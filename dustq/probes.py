"""Cos probes for the quantized-Dust study (Plan v3.1 section 5).

All probes operate on one fixed batch taken from the training pool (plus one held-out
val batch for directional descent), on a backprop(-STE) checkpoint trained at 1M tokens.
Sequence in the paper: 3 (STE/SSTE bias) first - it can rule out H2 before Tier B spends
its budget - then 2 (cosine ladders), 5 (sigma-Delta sweep), 4 (directional descent),
1 (reference gradient, disattenuation).

Implementation notes:
- Gradients are collected per site (module name) by running a step with an lr=0 SGD
  (DUST) or a normal backward (BP) and reading `.grad`; site grouping for reporting:
  attn.c_q/c_k/c_v -> Q/K/V, attn.c_proj -> Attn out, mlp.c_fc -> MLP in,
  mlp.c_proj -> MLP out, wte -> Tok emb, lm_head -> Head.
- The reference gradient is Dust at K=16384 split into two independent 8192 halves; the
  cosine between halves estimates the reference's own noise and disattenuates the
  measured cosines (cos_corrected = cos_measured / reliability).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RunConfig, load_config
from .upstream import ensure_upstream_on_path

ensure_upstream_on_path()

from dust import load_sequences  # noqa: E402

SITE_GROUPS = {
    "attn.c_q": "Q", "attn.c_k": "K", "attn.c_v": "V",
    "attn.c_proj": "Attn out", "mlp.c_fc": "MLP in", "mlp.c_proj": "MLP out",
    "transformer.wte": "Tok emb", "lm_head": "Head",
    # attention value-embedding internals. Before 2026-11-08 these fell through to their raw
    # parameter names ("transformer.h.1.attn.ve_gate.weight"), which (a) made every script
    # that filters on canonical groups silently drop them and (b) changed the number of
    # groups, hence the median `reliability` and every disattenuated number.
    "attn.ve_gate": "VE gate",
    "value_embeds": "Value emb",
}
# value_embeds is a top-level module indexed by layer (`value_embeds.{i}`), so it is a PREFIX
# of the parameter path where every other site is a suffix - group_of handles it separately.
_PREFIX_GROUPS = {"value_embeds": SITE_GROUPS["value_embeds"]}


def group_of(param_name: str) -> str:
    """Map a parameter name to its layer-type group (Plan v3.1 section 5). Parameter
    names carry a trailing `.weight`, so match against the module path with it stripped."""
    base = param_name[:-len(".weight")] if param_name.endswith(".weight") else param_name
    for prefix, name in _PREFIX_GROUPS.items():
        if base == prefix or base.startswith(prefix + "."):
            return name
    for key, name in SITE_GROUPS.items():
        if base == key or base.endswith("." + key):
            return name
    return param_name


def collect_grads(model) -> dict[str, torch.Tensor]:
    """Flatten per-parameter gradients into {name: flat fp32 tensor}."""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().float().flatten()
    return grads


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm().item() * b.norm().item()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def cosine_by_group(grads_a: dict, grads_b: dict) -> dict[str, float]:
    """Cosine per layer-type group, averaging per-parameter cosines within a group."""
    by_group: dict[str, list[float]] = {}
    for name, ga in grads_a.items():
        if name not in grads_b:
            continue
        by_group.setdefault(group_of(name), []).append(cosine(ga, grads_b[name]))
    return {g: sum(v) / len(v) for g, v in sorted(by_group.items())}


@dataclass
class ProbeContext:
    """Everything a probe run needs: the quantized model, its checkpoint, the batch.

    `batch` / `heldout` override the drive-loaded batches — for synthetic fixtures and CPU
    tests (the checkpoint dir is real; the data files may not exist)."""
    cfg: RunConfig
    device: torch.device = field(init=False)
    batch: tuple | None = None
    heldout: tuple | None = None

    def __post_init__(self):
        from dustq import qmodel
        from dustq.runner import build_model
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        from .train import load_best
        from dustq import qmodel as _qm  # noqa
        ckpt = Path(self.cfg.output_dir) / self.cfg.run_id / "best.pt"
        best, state = load_best(ckpt, self.device)
        # BP path: the same quantized model the checkpoint was trained under.
        self.model = build_model(self.cfg.model_kwargs(), self.cfg.precision,
                                 self.device, seed=self.cfg.seed).requires_grad_(True)
        from dustq.backprop_ext import make_bp_optimizer
        self.model.load_state_dict(state)  # latents are fp32; state is on-device
        # OQ-40: the Dust probe path must share the backprop probes' quantized forward
        # (Plan v3.1 4.2, one function per comparison). cfg.dust_forward: "fake" (the
        # plan-consistent default and 1.4x faster for the estimator) or "tf32"/"fp32"
        # (exact-path ablation). Without this, probe cosines would compare gradients
        # from two slightly different quantized functions.
        qmodel.set_exact_path(self.model, True, mode=self.cfg.dust_forward)
        self.optimizer = make_bp_optimizer(self.model, 1e-3, 0.9)
        self.best_val = best
        if self.batch is not None:
            self.x, self.y = self.batch
        elif self.cfg.data_dir == "synthetic":
            # local-only fallback, mirroring Runner._load_data: the same batch a synthetic
            # checkpoint was trained on. Without it the probe CLI could not be run off the
            # L4 at all (it demanded data/train_1M.pt), which is how the reporting crash
            # reached Colab untested. 8 sequences, as the drive path uses.
            from .runner import tiny_batch

            vocab = self.model.config.vocab_size
            length = self.model.config.sequence_len
            self.x, self.y = tiny_batch(self.device, seed=self.cfg.seed + 1,
                                        sequences=8, length=length, vocab=vocab)
        else:
            rows = load_sequences(Path(self.cfg.data_dir) / "train_1M.pt", 8)
            self.x = rows[:, :-1].contiguous().to(self.device)
            self.y = rows[:, 1:].contiguous().to(self.device)
        if self.heldout is not None:
            self.vx, self.vy = self.heldout
        elif self.cfg.data_dir == "synthetic":
            from .runner import tiny_batch

            vocab = self.model.config.vocab_size
            length = self.model.config.sequence_len
            self.vx, self.vy = tiny_batch(self.device, seed=90 + self.cfg.seed,
                                          sequences=8, length=length, vocab=vocab)
        else:
            val = load_sequences(Path(self.cfg.data_dir) / "val.pt", 8)
            self.vx = val[:, :-1].contiguous().to(self.device)
            self.vy = val[:, 1:].contiguous().to(self.device)

    # ---- gradient collection ----

    def _ensure_hooks(self, ste_mode: str = "identity", force: bool = False) -> None:
        """Activation-quant hooks must exist for EVERY gradient path — including the Dust
        paths (the estimator measures the QUANTIZED forward). Lazy registration here was a
        silent-correctness bug: dust_grads ran with quantized weights but float
        activations. Registered once per mode; the SMOKE task keeps absmax (no alpha table).

        `force` re-registers even when the mode is unchanged: on the Dust path the hooks
        must land AFTER upstream's o@l noise hook on EVERY call (DUSTQ.close() removes
        Dust's hooks but leaves ours in place, so a cached registration order would put us
        back in front of the next estimator's noise)."""
        if not force and getattr(self, "_hook_mode", None) == ste_mode \
                and getattr(self, "_hook_handles", None):
            return
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        from dustq import qmodel
        if self.cfg.precision != "fp":
            _, bits, scheme = qmodel.parse_precision(self.cfg.precision)
            if self.cfg.task == "smoke" and scheme == "mse_alpha":
                scheme = "absmax"
            self._hook_handles = qmodel.register_act_hooks(
                self.model, bits, scheme,
                allow_lowbit_absmax=(self.cfg.task == "smoke"),
                ste_mode=ste_mode, n_layer=self.model.config.n_layer)
        else:
            self._hook_handles = []
        self._hook_mode = ste_mode

    def dust_grads(self, seed: int, population: int = 256, sigmas: dict | None = None,
                   apply_update: bool = False) -> dict:
        """A Dust gradient estimate at `population` (seed selects the noise stream)."""
        from .recipes import recipe
        from .dust_ext import DUSTQ
        model = self.model
        model.requires_grad_(True)
        settings = recipe(self.cfg.tokens, population, model.config.n_layer)
        # Hook order is load-bearing (qmodel.py header, Runner._build): construct the
        # estimator FIRST, then attach the act-quant hooks, giving noise -> quantize.
        # The probes used to do this in the opposite order, so the estimator's jitter was
        # applied to an already-quantized input (and on the tf32/fp32 exact paths it was
        # dropped entirely, because QLinear's codes are computed from the clean input the
        # act hook sees). That made every Dust probe gradient a different function from
        # the one Tier B trains.
        est = DUSTQ(model, settings, seed=seed, sigmas=sigmas)
        self._ensure_hooks("identity", force=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.0 if not apply_update else 1e-3)
        est.step(self.x, self.y, opt)
        grads = collect_grads(model)
        est.close()
        model.requires_grad_(True)
        return grads

    def bp_grads(self, ste_mode: str = "identity") -> dict:
        """A backprop gradient at the same weights (BP-STE or BP-SSTE)."""
        from dustq.backprop_ext import train_step  # noqa: F401  (import kept for parity)
        # fresh hooks per backward mode so BOTH modes evaluate on the SAME weights
        self._ensure_hooks(ste_mode)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.model_device_str(), dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            loss = self.model(self.x, self.y)
        loss.backward()
        grads = collect_grads(self.model)
        return grads

    def model_device_str(self) -> str:
        return self.device.type

    # ---- losses ----

    def quantized_losses(self, x, y) -> torch.Tensor:
        self._ensure_hooks(self._hook_mode or "identity")
        with torch.no_grad():
            with torch.autocast(self.model_device_str(), dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                return self.model(x, y, loss_reduction="none")


# ---------- the five probes ----------

def probe_reference(ctx, k_total: int = 16384) -> tuple[dict, dict, float]:
    """Probe 1: Dust gradient at K=16k split into two independent halves at the largest
    recipe-preset population <= half (100k-token demos needs recipe presets: upstream
    `recipe()` only accepts {256, 1024, 4096, 16384}); returns (g1, g2, reliability)
    where reliability = median per-group cosine between halves (disattenuation)."""
    import bisect
    half = k_total // 2
    # largest preset <= half (fallback: largest supported)
    half = max(p for p in (256, 1024, 4096) if p <= half) if any(p <= half for p in (256, 1024, 4096)) else 1024
    g1 = ctx.dust_grads(seed=0, population=half)
    g2 = ctx.dust_grads(seed=5000, population=half)
    rel = sorted(cosine_by_group(g1, g2).values())
    reliability = rel[len(rel) // 2]  # median across groups
    return g1, g2, reliability


def probe_averaging_test(ctx, ns=(256, 1024, 4096), trials: int = 5, seed: int = 0) -> dict:
    """Watch lead (2026-10-31, OPEN_QUESTIONS 37; cf. Apical Dust's Averaging Test):
    per-tensor estimator BIAS vs population N, on a fixed batch. For each N, run
    `trials` independent estimates g_i and report, per tensor,
    mean_i <g_i, g_ref> - |g_ref|^2 / 2 ... i.e. the bias component along the reference
    direction: bias_N = mean_i <g_i, g_ref>/||g_ref|| - ||g_ref||/2. A non-decaying
    bias_N at large N is systematic estimator bias that the cosine ladder cannot see
    (cosine measures alignment only; this measures offset along the true direction).
    Honest limitation: DUSTQ.step() does not expose single-draw estimates, so the test
    uses `trials` independent K=N estimates; single-N draws would sharpen it but need an
    upstream hook (recorded in OPEN_QUESTIONS). Runs entirely without training."""
    ref = ctx.dust_grads(seed=seed, population=16384 // 4)  # 4096 preset reference
    ref_norm = {g: r.norm().item() for g, r in ref.items()}
    out = {}
    for n in ns:
        dots, devs = {}, {}
        for t in range(trials):
            g = ctx.dust_grads(seed=seed + t * 5000, population=n)
            for name, r in ref.items():
                if name not in g:
                    continue
                dots.setdefault(name, []).append(torch.dot(g[name], r).item())
                devs.setdefault(name, []).append(g[name] - r)
        out[n] = {name: {
            "bias_along_ref": sum(dots[name]) / trials / max(ref_norm[name], 1e-12) - ref_norm[name] / 2,
            "rms_dev": torch.stack([d.norm() for d in devs[name]]).mean().item(),
        } for name in dots}
    return out


def probe_per_tensor_cosine(ctx, reference, populations=(256, 1024)) -> dict:
    """Watch lead (2026-10-31, OPEN_QUESTIONS 37; cf. the Noise fork's check_quant.py):
    the same cosines as probe 2 but per PARAMETER TENSOR rather than per group — the
    fork's harness is per-tensor, so this cross-checks our grouped numbers against
    the external convention. Returns {population: {param_name: cos}}."""
    out = {}
    for k in populations:
        g = ctx.dust_grads(seed=0, population=k)
        out[k] = {name: cosine(g[name], reference[name])
                  for name in reference if name in g}
    return out


def probe_seed_noise(ctx, seeds=(0, 1), populations=(256, 1024)) -> dict:
    """Probe 0 (seed-pair noise floor): per-group cosine between Dust gradients at two
    seeds on the SAME checkpoint and batch. Benchmarks any "within seed noise" claim: the
    G1 dtype verdict was a mean cosine of +0.9910 (fp32 vs bf16); if seed-vs-seed is at or
    below that, the two dtypes are indistinguishable at this K."""
    out = {}
    for k in populations:
        g0 = ctx.dust_grads(seed=seeds[0], population=k)
        g1 = ctx.dust_grads(seed=seeds[-1], population=k)
        out[str(k)] = cosine_by_group(g0, g1)
    return out


def probe_cosine_ladders(ctx, populations=(256, 1024, 4096), reference=None,
                         reliability: float | None = None) -> dict:
    """Probe 2: cos(Dust-K, ref) per layer type vs K. Reports BOTH the raw cosine and the
    disattenuated one (raw / halves-reliability) per the reporting precedent
    arXiv:2609.19589 (WATCH 2026-11-01B): state what you divided out. Returns
    {population: {group: {"raw": c, "disattenuated": min(1, c/reliability)}}}.

    `reliability` is the median per-group cosine between the reference's two halves. It is
    only computed here when the caller did not already have it (which costs another
    K=4096 pair). Passing it is not cosmetic: the CLI used to leave it at 1.0, so every
    "disattenuated" number in the shipped JSONs equalled the raw one while the JSON also
    reported a reliability of 0.29-0.38 - i.e. the field looked applied and was not."""
    if reference is None:
        _, reference, computed = probe_reference(ctx)
        if reliability is None:
            reliability = computed
    elif reliability is None:
        reliability = 1.0
    out = {}
    for k in populations:
        g = ctx.dust_grads(seed=0, population=k)
        per_group = cosine_by_group(g, reference)
        out[k] = {gname: {"raw": c,
                          # clamp BOTH ends: a cosine outside [-1, 1] is meaningless, and the
                          # old min(1, ...) left negative values unbounded (-1.45 appeared in
                          # the M data, where reliability is only 0.08)
                          "disattenuated": max(-1.0, min(1.0, c / reliability))}
                  for gname, c in per_group.items()}
    return out


def probe_ste_bias(ctx, reference=None) -> dict:
    """Probe 3 (runs first): cos(STE, ref) and cos(SSTE, ref) per bit-width/group. Cheap
    gate on H2: if cos(STE, ref) already equals cos(fp-BP, fp-ref) at the primary bit
    width, STE bias is not the mechanism and Tier B re-decides its budget."""
    if reference is None:
        _, reference, _ = probe_reference(ctx)
    ste = ctx.bp_grads(ste_mode="identity")
    sste = ctx.bp_grads(ste_mode="smoothed")
    return {
        "ste": cosine_by_group(ste, reference),
        "sste": cosine_by_group(sste, reference),
    }


def directional_descent(evaluate, steepest=1e-1, steps=12):
    """Probe 4 helper: log-grid line search around eps=1, returns (best_delta, eps)."""
    import numpy as np

    grid = [2.0 ** e for e in np.linspace(math.log2(steepest) - steps,
                                          math.log2(steepest) + steps, 2 * steps + 1)]
    best = (float("inf"), 0.0)
    for eps in grid:
        value = evaluate(eps)
        if value < best[0]:
            best = (value, eps)
    return best


def probe_directional_descent(ctx, candidates: dict, grid: int = 12) -> dict:
    """Probe 4: for each candidate direction, min over eps of L_q(theta - eps*d/||d||)
    - L_q(theta), on the training batch and one held-out val batch. L_q is the real
    quantized loss, so this is the method-neutral arbiter."""
    import copy

    from .quant import ste_act_quantize  # noqa: F401  (documentation import)

    results = {}
    base_train = ctx.quantized_losses(ctx.x, ctx.y).mean().item()
    base_val = ctx.quantized_losses(ctx.vx, ctx.vy).mean().item()
    for name, grads in candidates.items():
        flat = torch.cat([g for g in grads.values()])
        norm = flat.norm().item()

        def ev(eps, direction=grads, batch=ctx.x, target=ctx.y, which="train"):
            with torch.no_grad():
                backup = [p.detach().clone() for p in ctx.model.parameters()]
                for (pname, param) in list(ctx.model.named_parameters()):
                    if pname in direction and eps:
                        param.add_(direction[pname].reshape(param.shape).to(param.device) * (-eps / max(direction[pname].norm().item(), 1e-12)))
                value = ctx.quantized_losses(batch, target).mean().item()
                for p, b in zip(ctx.model.parameters(), backup):
                    p.copy_(b)
            return value - (base_train if which == "train" else base_val)

        results[name] = {
            "train": directional_descent(lambda e: ev(e, which="train")),
            "val": directional_descent(lambda e: ev(e, which="val")),
            "norm": norm,
        }
    return results


def fit_cos_law(cos: dict[int, float]):
    """Fit cos(K) = c_max / sqrt(1 + c/K) per group (Plan v3.1 Appendix G law)."""
    import numpy as np

    out = {}
    for group, pts in cos.items():
        ks = sorted(pts)
        if len(ks) < 3:
            out[group] = None
            continue
        y = np.array([pts[k] for k in ks], dtype=float)
        x = np.array([1.0 / k for k in ks], dtype=float)
        # y = c_max / sqrt(1 + c*x): for a given c, fit c_max = argmin ||y - c_max*a||^2
        best = None
        for c in np.geomspace(1e-1, 1e8, 60):
            a = 1.0 / np.sqrt(1.0 + c * x)
            cmax = float(a @ y / (a @ a))
            rmse = float(np.sqrt(((cmax * a - y) ** 2).mean()))
            if best is None or rmse < best[1]:
                best = (cmax, rmse, float(c))
        out[group] = {"c_max": best[0], "c": best[2], "rmse": best[1]}
    return out


# ---------- probe 5: sigma-Delta ----------

def median_delta(scales) -> float:
    """Median per-token grid step Delta. For our quantizer the code spacing IS the
    per-token scale (codes are integers; dequant = code x scale), so Delta = median(scale).
    The previous /255 was an 8-bit absmax artifact and is meaningless on the A4/A2
    mse_alpha grids this sweep runs on."""
    return float(scales.detach().float().median().item())


def probe_sigma_delta(ctx, reference, sigma_fractions=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16),
                      populations=(256, 1024)) -> dict:
    """Probe 5: hold Dust's jitter at sigma = frac x the site's grid step Delta and measure
    what survives. Per (frac, K): per-group cosine to the backprop reference (same grouping
    as probes 2/3). Per frac: the activation-code FLIP RATE — the fraction of codes a
    single Dust-scale draw moves off their clean value — which is the direct dead-zone /
    smoothing diagnostic the sweep exists to expose.

    Sweep design [design]: sigma is scaled jointly for the jitter families that have an
    activation-quant counterpart (local, hub_shallow/hub_deep, hidden, writer — the
    inverse of qmodel.site_sigma_key); emb/head/scalar keep their upstream defaults. So
    sigma = frac x Delta means "one draw's std, in units of one code step" (1x = noise the
    width of a code, 16x = noise crossing many). Cross-tensor caveat [design]: only
    attn.c_proj's input is both jittered (Dust's o@l pre-hook) and quantized (our hook);
    for the other three families Dust jitters a neighbouring tensor of the same site, so
    the sigma-vs-Delta ratio compares nearby tensors rather than identical ones.

    Flip rate is a property of the quantizer and the jitter only — independent of the
    matmul mode and of K — so it is computed once per frac and repeated for each K."""
    from torch import nn

    from dustq import qmodel

    ctx._ensure_hooks("identity")  # the spies must attach AFTER the act-quant pre-hooks
    _, bits, _ = qmodel.parse_precision(ctx.cfg.precision)

    # Spy pre-hooks (chained after the act hooks, so they see the quantized input) capture
    # the clean codes, the per-token scale and a bounded slice of the raw input per site.
    clean: dict[str, dict] = {}

    def spy(name):
        def hook(module, args):
            if getattr(module, "_act_codes", None) is None:
                return None
            x = args[0]
            if x.is_floating_point():
                # two sequences only: flips need a bounded sample, and this keeps the big
                # tensors off the GPU for the whole sweep
                clean[name] = {
                    "input": x.detach()[:2].clone(),
                    "codes": module._act_codes.detach()[:2].clone(),
                    "scale": module._act_scale.detach().clone(),
                }
            return None
        return hook

    handles = [module.register_forward_pre_hook(spy(name))
               for name, module in ctx.model.named_modules()
               if qmodel._site_matches(name, qmodel.ACT_QUANT_SITES)
               and isinstance(module, nn.Linear)]
    was_training = ctx.model.training
    ctx.model.eval()
    with torch.no_grad():
        ctx.model(ctx.x, ctx.y)
    for h in handles:
        h.remove()
    ctx.model.train(was_training)

    # one Delta per Dust sigma key, averaged over the sites that map to it
    n_layer = ctx.model.config.n_layer
    per_key: dict[str, list[float]] = {}
    for name, data in clean.items():
        key = qmodel.site_sigma_key(name, n_layer)
        if key is not None:
            per_key.setdefault(key, []).append(median_delta(data["scale"]))
    delta = {k: sum(v) / len(v) for k, v in sorted(per_key.items())}

    def flip_rates(sigmas: dict) -> dict:
        """Fraction of clean activation codes that one Dust-scale draw moves, per group."""
        rates = {}
        for name, data in clean.items():
            key = qmodel.site_sigma_key(name, n_layer)
            sigma = sigmas.get(key)
            if key is None or sigma is None:
                continue
            x, codes, scale = data["input"], data["codes"], data["scale"]
            scale = scale[: x.shape[0]]  # same leading window as the sliced codes/input
            if qmodel.site_unsigned(name):
                lo, hi = 0, 2 ** bits - 1
            else:
                lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(7))
            noise = noise.to(x.device, x.dtype).mul_(sigma)
            moved = (torch.clamp(torch.round((x + noise) / scale), lo, hi) != codes)
            rates[group_of(name + ".weight")] = float(moved.float().mean().item())
        return {g: rates[g] for g in sorted(rates)}

    out = {}
    for frac in sigma_fractions:
        sigmas = {k: frac * d for k, d in delta.items()}
        flips = flip_rates(sigmas)
        for k in populations:
            g = ctx.dust_grads(seed=0, population=k, sigmas=sigmas)
            out[(frac, k)] = {
                "cos": cosine_by_group(g, reference),
                "flip_rate": flips,
                "sigma": {key: round(val, 6) for key, val in sigmas.items()},
            }
    return out


# ---------- CLI (one cell on Colab) ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML of the checkpoint run (PCKPT)")
    parser.add_argument("--out", default="results/probes")
    parser.add_argument("--probes", default="3,2,5,4",
                        help="subset of {0,1,2,3,4,5}; probe 1 is computed once for any that need it")
    parser.add_argument("--dust-forward", default=None, choices=["tf32", "fp32", "fake"],
                        help="override the config's dust_forward (OQ-40) for this probe run")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if args.dust_forward:
        cfg.dust_forward = args.dust_forward
    ctx = ProbeContext(cfg)
    which = {int(p) for p in args.probes.split(",")}
    report = {}

    def save() -> None:
        """Write the report so far. Called after every probe so a later crash (or an L4
        session timeout) cannot take the completed, expensive probes' results with it.

        The filename carries the probe set: the session script runs each subset as its own
        subprocess, and a fixed name meant the later subsets silently overwrote the earlier
        ones' JSON (probe 5's write erased the ladder and the descent that G2a needs).
        """
        tag = "-".join(str(p) for p in sorted(which))
        path = Path(args.out) / f"{cfg.run_id}_probes_{tag}.json"
        path.write_text(json.dumps(report, indent=1))
        print("wrote", path, flush=True)

    if 0 in which:
        print("probe 0: seed-pair noise floor", flush=True)
        noise = probe_seed_noise(ctx)
        report["seed_noise"] = noise
        save()
        print(json.dumps(noise, indent=1))
        flat = [v for groups in noise.values() for v in groups.values()]
        if flat:
            print(f"seed-noise floor: mean {sum(flat) / len(flat):.4f}, "
                  f"min {min(flat):.4f} — compare with the G1 bf16-vs-fp32 mean (+0.9910)", flush=True)

    ref, reliability = None, 1.0
    if which & {1, 2, 3, 5}:
        _, ref, reliability = probe_reference(ctx)
        report["reliability"] = reliability
        save()
        print(f"probe 1: reference K=16384 halves -> reliability {reliability:.4f}", flush=True)

    if 2 in which:
        print("probe 2: cosine ladders", flush=True)
        ladders = probe_cosine_ladders(ctx, reference=ref, reliability=reliability)
        report["cosine_ladders"] = {str(k): v for k, v in ladders.items()}
        report["disattenuation_applied"] = reliability
        save()
        print(json.dumps(report["cosine_ladders"], indent=1))

    if 3 in which:
        print("probe 3: STE/SSTE bias", flush=True)
        bias = probe_ste_bias(ctx, reference=ref)
        report["ste_bias"] = bias
        save()
        print(json.dumps(bias, indent=1))

    if 5 in which:
        print("probe 5: sigma-Delta sweep", flush=True)
        sweep = probe_sigma_delta(ctx, ref)
        fracs = sorted({key[0] for key in sweep})
        ks = sorted({key[1] for key in sweep})
        report["sigma_delta"] = {
            # flip rate and sigma do not vary with K; report them once per frac
            "cos": {f"{k[0]:.3f}x@{k[1]}": v["cos"] for k, v in sweep.items()},
            "flip_rate": {f"{f:.3f}x": sweep[(f, ks[0])]["flip_rate"] for f in fracs},
            "sigma": {f"{f:.3f}x": sweep[(f, ks[0])]["sigma"] for f in fracs},
        }
        # Persist BEFORE pretty-printing. The 2026-11-02 session lost a full 16-point sweep
        # because a KeyError in the table (sweep[(frac, k)] with k already a (frac, K) key)
        # fired after the expensive work and before the JSON write. Data first, cosmetics
        # second — and every probe saves as it goes.
        save()
        print("  frac   K      mean_cos   flip_rate (per group)")
        for frac in fracs:
            for k in ks:
                cos = sweep[(frac, k)]["cos"]
                flips = sweep[(frac, k)]["flip_rate"]
                mean_cos = sum(cos.values()) / max(len(cos), 1)
                mean_flip = sum(flips.values()) / max(len(flips), 1)
                detail = " ".join(f"{g}={v:.3f}" for g, v in flips.items())
                print(f"  {frac:5.3f}  {k:<5d}  {mean_cos:+8.4f}   "
                      f"{mean_flip:.4f}  | {detail}")
        if 1.0 in fracs:
            print("sigma per key at 1x:", json.dumps(sweep[(1.0, ks[0])]["sigma"]))
        else:
            print("(no 1.0 point; deltas", json.dumps(sweep[(fracs[0], ks[0])]["sigma"]), ")")

    if 4 in which:
        print("probe 4: directional descent", flush=True)
        cand = {
            "STE": ctx.bp_grads("identity"),
            "SSTE": ctx.bp_grads("smoothed"),
            "Dust256": ctx.dust_grads(seed=0, population=256),
            "Dust1k": ctx.dust_grads(seed=0, population=1024),
        }
        dd = probe_directional_descent(ctx, cand)
        report["directional_descent"] = {k: {"train": v["train"], "val": v["val"]}
                                         for k, v in dd.items()}
        save()
        print(json.dumps(report["directional_descent"], indent=1))

    save()


if __name__ == "__main__":
    main()


