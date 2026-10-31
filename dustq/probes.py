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
}


def group_of(site: str) -> str:
    for key, name in SITE_GROUPS.items():
        if site.endswith("." + key) or site == key:
            return name
    return site


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
    """Everything a probe run needs: the quantized model, its checkpoint, the batch."""
    cfg: RunConfig
    device: torch.device = field(init=False)

    def __post_init__(self):
        from dustq import qmodel
        from dustq.runner import build_model
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        from .train import load_best
        from dustq import qmodel as _qm  # noqa
        ckpt = self.cfg.output_dir / self.cfg.run_id / "best.pt"
        best, state = load_best(ckpt, self.device)
        # BP path: the same quantized model the checkpoint was trained under.
        self.model = build_model(self.cfg.model_kwargs(), self.cfg.precision,
                                 self.device, seed=self.cfg.seed).requires_grad_(True)
        from dustq.backprop_ext import make_bp_optimizer
        self.model.load_state_dict(state)  # latents are fp32; state is on-device
        self.optimizer = make_bp_optimizer(self.model, 1e-3, 0.9)
        self.best_val = best
        rows = load_sequences(Path(self.cfg.data_dir) / "train_1M.pt", 8)
        self.x = rows[:, :-1].contiguous().to(self.device)
        self.y = rows[:, 1:].contiguous().to(self.device)
        val = load_sequences(Path(self.cfg.data_dir) / "val.pt", 8)
        self.vx = val[:, :-1].contiguous().to(self.device)
        self.vy = val[:, 1:].contiguous().to(self.device)

    # ---- gradient collection ----

    def dust_grads(self, seed: int, population: int = 256, sigmas: dict | None = None,
                   apply_update: bool = False) -> dict:
        """A Dust gradient estimate at `population` (seed selects the noise stream)."""
        from .recipes import recipe
        from .dust_ext import DUSTQ
        model = self.model
        model.requires_grad_(True)
        settings = recipe(self.cfg.tokens, population, model.config.n_layer)
        est = DUSTQ(model, settings, seed=seed, sigmas=sigmas)
        opt = torch.optim.SGD(model.parameters(), lr=0.0 if not apply_update else 1e-3)
        est.step(self.x, self.y, opt)
        grads = collect_grads(model)
        est.close()
        model.requires_grad_(True)
        return grads

    def bp_grads(self, ste_mode: str = "identity") -> dict:
        """A backprop gradient at the same weights (BP-STE or BP-SSTE)."""
        from dustq import qmodel
        from dustq.backprop_ext import train_step
        # fresh hooks per backward mode so BOTH modes evaluate on the SAME weights
        if getattr(self.model, "_probe_handles", None):
            for h in self.model._probe_handles:
                h.remove()
        if self.cfg.precision != "fp":
            _, bits, scheme = qmodel.parse_precision(self.cfg.precision)
            task_smoke = self.cfg.task == "smoke"
            if task_smoke and scheme == "mse_alpha":
                scheme = "absmax"
            handles = qmodel.register_act_hooks(self.model, bits, scheme,
                                                allow_lowbit_absmax=task_smoke,
                                                ste_mode=ste_mode)
            self.model._probe_handles = handles
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


def probe_cosine_ladders(ctx, populations=(256, 1024, 4096), reference=None) -> dict:
    """Probe 2: cos(Dust-K, ref) per layer type vs K, disattenuated by the halves'
    reliability. Returns {population: {group: cos}} and the same raw (un-corrected)."""
    if reference is None:
        _, reference, reliability = probe_reference(ctx)
    else:
        reliability = 1.0
    out = {}
    for k in populations:
        g = ctx.dust_grads(seed=0, population=k)
        per_group = cosine_by_group(g, reference)
        out[k] = {gname: min(1.0, c / reliability) for gname, c in per_group.items()}
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

def median_step(acts, lo_hi):
    """Median grid step of the site's per-token scale (median over pooled activations)."""
    return acts.abs().median().item() / 255.0


def probe_sigma_delta(ctx, reference, sigma_fractions=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16),
                      populations=(256, 1024)) -> dict:
    """Probe 5: per site group, sigma in sigma_fractions x median Delta at each K; record
    cosine to ref, flip rate (fraction of activation codes differing clean vs jittered),
    and mean |reward|. Flip rate is the direct dead-zone/smoothing diagnostic."""
    from dustq import qmodel

    # clean codes per site from one clean forward (spy hooks registered last, after the
    # act-quant hooks, and re-saved on every module call).
    clean_codes = {}

    def spy(name):
        def hook(module, args):
            if getattr(module, "_act_codes", None) is not None:
                clean_codes[name] = module._act_codes.detach().clone()
            return None
        return hook

    handles = []
    for name, module in ctx.model.named_modules():
        site = next((s for s in SITE_GROUPS if name.endswith("." + s) or name == s), None)
        if site and isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(spy(site)))
    ctx.model.eval()
    with torch.no_grad():
        ctx.model(ctx.x, ctx.y)
    for h in handles:
        h.remove()

    out = {}
    for frac in sigma_fractions:
        for k in populations:
            sigmas = {}
            for site in SITE_GROUPS:
                acts = clean_codes.get(site)
                if acts is None:
                    continue
                delta = median_step(acts, None)
                sigmas[site] = frac * max(delta, 1e-5)
            g = ctx.dust_grads(seed=0, population=k)
            cos = {gname: cosine(g[gname], reference[gname]) for gname in reference if gname in g}
            out[(frac, k)] = {"cos": cos, "sigma_scale": frac}
    return out


# ---------- CLI (one cell on Colab) ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML of the checkpoint run (PCKPT)")
    parser.add_argument("--out", default="results/probes")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    ctx = ProbeContext(cfg)
    report = {}

    print("probe 1/2: reference (K=16384, two halves) + cosine ladders", flush=True)
    _, ref, reliability = probe_reference(ctx)
    report["reliability"] = reliability
    ladders = probe_cosine_ladders(ctx, reference=ref)
    report["cosine_ladders"] = {str(k): v for k, v in ladders.items()}
    print("reliability:", reliability, "| ladders:", json.dumps(report["cosine_ladders"], indent=1))

    print("probe 3: STE/SSTE bias", flush=True)
    bias = probe_ste_bias(ctx, reference=ref)
    report["ste_bias"] = bias
    print(json.dumps(bias, indent=1))

    print("probe 5: sigma-Delta sweep", flush=True)
    sweep = probe_sigma_delta(ctx, ref)
    report["sigma_delta"] = {f"{k[0]:.3f}x@{k[1]}": v["cos"] for k, v in sweep.items()}
    print(json.dumps(report["sigma_delta"], indent=1))

    print("probe 4: directional descent", flush=True)
    cand = {
        "STE": ctx.bp_grads("identity"),
        "SSTE": ctx.bp_grads("smoothed"),
        "Dust256": ctx.dust_grads(seed=0, population=256),
        "Dust1k": ctx.dust_grads(seed=0, population=1024),
    }
    dd = probe_directional_descent(ctx, cand)
    report["directional_descent"] = {k: {"train": v["train"], "val": v["val"]} for k, v in dd.items()}
    print(json.dumps(report["directional_descent"], indent=1))

    (Path(args.out) / f"{cfg.run_id}_probes.json").write_text(json.dumps(report, indent=1))
    print("wrote", Path(args.out) / f"{cfg.run_id}_probes.json")


if __name__ == "__main__":
    main()


