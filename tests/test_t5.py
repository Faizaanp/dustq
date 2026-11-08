"""T5 tests (Plan v3.1 section 7): the quantization module's full contract.

Plan tests 3, 4, 5, 7, 8 live here (tests 1, 2, 6 live in test_phase0), plus the T5
additions: code-cache invalidation (budget-critical), the fp32-latent assertion
(arXiv:2610.07853), SSTE-forward-equals-STE, the DUST-only exact path, and BP-SSTE smoke.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dustq import qmodel, runner
from dustq.dust_ext import DUSTQ
from dustq.quant import (QLinear, _SSTEActQuant, sste_act_quantize, ste_act_quantize,
                         swap_linear)
from dustq.upstream import ensure_upstream_on_path

ensure_upstream_on_path()

from dust import DUST, Recipe  # noqa: E402


def tiny_recipe():
    """Hand-built valid recipe for the tiny model (all counts divisible by nrep=2)."""
    return Recipe(writers=[2], hidden=[2], hubs=[2], embedding=2, head=32,
                  local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=0.1, momentum=0.9)


# ---------- T5 test 7: SSTE backward vs Monte-Carlo ----------

def _sste_mc_forward(x, delta, sigma, lo, hi, n=200_000, seed=0):
    """E[clamp(round((x + sigma*e)/Delta)) * Delta] - the DEQUANTIZED forward, i.e. exactly
    what _SSTEActQuant.forward returns, noise-averaged. The original test differentiated
    round(x/Delta) (the code), which shares the implementation's 1/Delta factor and so could
    not see it."""
    g = torch.Generator().manual_seed(seed)
    q = torch.clamp(torch.round((x.unsqueeze(0) + sigma * torch.randn(n, x.numel(), generator=g)) / delta),
                    lo, hi) * delta
    return q.mean(0)


def test_sste_backward_matches_mc():
    """Monte-Carlo agreement at BOTH regimes the old +/-3 test never separated:
    sigma/Delta ~ 1 (valid) and sigma/Delta >> 1 (the A8 regime that collapsed P2)."""
    torch.manual_seed(0)
    lo, hi = -8, 7
    x = torch.tensor([0.0, 0.025, 0.02, 0.05, 0.30, -0.025, -0.10], requires_grad=True)
    x0 = x.detach()

    # sigma/Delta = 1: the regime the original test covered
    y = _SSTEActQuant.apply(x, torch.full_like(x, 0.05), lo, hi, 0.05)
    y.sum().backward()
    gx = x.grad.clone()
    h = 0.002
    xp, xm = x0.clone(), x0.clone()
    mc = (_sste_mc_forward(xp + h, 0.05, 0.05, lo, hi)
          - _sste_mc_forward(xm - h, 0.05, 0.05, lo, hi)) / (2 * h)
    assert torch.allclose(gx, mc, atol=0.25), (gx, mc)

    # sigma/Delta = 20: A8 grid with Dust's default sigma, where the old code kept ~14%
    x = torch.tensor([0.0, 0.0042, -0.0037, 0.0099], requires_grad=True)
    x0 = x.detach()
    y = _SSTEActQuant.apply(x, torch.full_like(x, 0.010), -128, 127, 0.2)
    y.sum().backward()
    gx = x.grad.clone()
    xp, xm = x0.clone(), x0.clone()
    mc = (_sste_mc_forward(xp + 1e-4, 0.010, 0.2, -128, 127)
          - _sste_mc_forward(xm - 1e-4, 0.010, 0.2, -128, 127)) / (2e-4)
    ratio = (gx.abs().sum() / mc.abs().sum()).item()
    assert 0.8 < ratio < 1.25, (
        f"at sigma/Delta=20 the boundary sum keeps {ratio:.3f} of the true derivative "
        f"(1.0 expected); analytic {gx.abs().sum():.4f} vs mc {mc.abs().sum():.4f}")
    # at sigma/Delta = 20 the smoothing is far wider than the grid, so the derivative is
    # essentially flat at 1 (every point is effectively at a boundary) - not peaked.
    assert (gx - 1.0).abs().max() < 0.05, (
        f"the wide-sigma derivative should be ~1 everywhere, got {gx.tolist()}")


def test_sste_backward_scale_is_delta_not_one_over_delta():
    """Guard the 1/Delta factor directly: with a fixed sigma, narrowing the grid (smaller
    Delta) must NOT inflate the gradient. Before the 2026-11-06 fix the backward returned the
    derivative of the CODE index, i.e. dQ/dx / Delta, which grew as Delta shrank."""
    x = torch.tensor([0.0, 0.03, -0.07, 0.21], requires_grad=True)
    means = []
    for delta in (0.05, 0.01, 0.002):
        if x.grad is not None:
            x.grad = None
        y = _SSTEActQuant.apply(x, torch.full_like(x, delta), -128, 127, 0.05)
        y.sum().backward()
        means.append(x.grad.mean().item())
    # the smoothed quantizer's average slope is ~1 at any Delta; it must not scale as 1/Delta
    for m in means:
        assert 0.5 < m < 1.5, f"mean dQ/dx = {m:.4f} is not the dequantized derivative"
    assert max(means) / min(means) < 2.0, (
        f"gradient magnitude varies {max(means)/min(means):.2f}x across Delta - "
        f"the 1/Delta factor is back: {means}")


def test_sste_forward_equals_ste_forward():

    torch.manual_seed(1)
    x = torch.randn(64, 128) * 3
    assert torch.equal(ste_act_quantize(x, 4), sste_act_quantize(x, 4, sigma=0.2))
    assert torch.equal(ste_act_quantize(x, 8, unsigned=True),
                       sste_act_quantize(x, 8, unsigned=True, sigma=0.4))


# ---------- T5 test 3: hook order (noise -> quantize) at o@l ----------

def test_hook_order_noise_then_quantize():
    model = runner.build_model(precision="w1.58a4", seed=0).requires_grad_(False)
    est = DUSTQ(model, tiny_recipe(), seed=0)
    spy = {"raw": None, "post": None}

    def spy_hook(_module, args):
        spy["raw"] = args[0].detach().clone()  # AFTER the noise hook, BEFORE quant
        return None

    def spy_post_hook(_module, args):
        spy["post"] = args[0].detach().clone()  # AFTER the quant hook

    model.transformer.h[0].attn.c_proj.register_forward_pre_hook(spy_hook)
    qmodel.register_act_hooks(model, 4, "absmax", allow_lowbit_absmax=True)
    model.transformer.h[0].attn.c_proj.register_forward_pre_hook(spy_post_hook)
    qmodel.set_exact_path(model, True)

    x, y = runner.tiny_batch(torch.device("cpu"), seed=5)
    sigma = qmodel.site_sigma("transformer.h.0.attn.c_proj", model.config.n_layer)  # 0.2 shallow
    est.noise = lambda *shape, dtype=None: torch.ones(*shape, device=x.device)  # fixed noise
    with torch.no_grad():
        est.capture(x, y)
        est.direct_errors(["o@0"], 2, sigma)["o@0"]

    # the draw's forward passed through noise -> quantize in that order: the post-quant
    # value must equal quantize(raw-noisy-input). If the order were flipped, the post
    # value would be dequant(quantize(o)) + sigma instead.
    expected = ste_act_quantize(spy["raw"], 4, scheme="absmax")
    assert spy["post"].shape == expected.shape
    assert torch.allclose(spy["post"], expected), "noise must precede quantization at o@l"


# ---------- T5 test 4: captured input = dequantized quantized input ----------

def test_captured_input_is_dequantized():
    model = runner.build_model(precision="w1.58a4", seed=0).requires_grad_(False)
    est = DUST(model, tiny_recipe(), seed=0)
    qmodel.register_act_hooks(model, 4, "absmax", allow_lowbit_absmax=True)
    x, y = runner.tiny_batch(torch.device("cpu"), seed=7)
    with torch.no_grad():
        est.capture(x, y)
    captured = est.inputs["transformer.h.0.attn.c_q"]
    # manual: c_q's input is upstream of every quantizer -> exact recompute
    from model import norm
    x0 = norm(model.transformer.wte(x))
    xin = model.resid_lambdas[0] * x0 + model.x0_lambdas[0] * x0
    expected = ste_act_quantize(norm(xin), 4, scheme="absmax")
    assert torch.allclose(captured, expected)


# ---------- T5 test 5: zero-noise DUSTQ forward == quantized forward ----------

def test_zero_noise_dustq_forward_equals_quantized_forward():
    model = runner.build_model(precision="w1.58a4", seed=0).requires_grad_(False)
    est = DUSTQ(model, tiny_recipe(), seed=0)
    qmodel.register_act_hooks(model, 4, "absmax", allow_lowbit_absmax=True)
    qmodel.set_exact_path(model, True)
    x, y = runner.tiny_batch(torch.device("cpu"), seed=9)
    with torch.no_grad():
        est.capture(x, y)
        direct = model(x, y, loss_reduction="none").view_as(y).float()
    assert torch.equal(est.clean_loss, direct), "capture path diverged from the quantized forward"


# ---------- T5 test 8: integer-code matmul vs fp64 reference ----------

def test_code_matmul_matches_fp64():
    torch.manual_seed(2)
    q = QLinear(32, 16, mode="ternary")
    q.weight.data.normal_()
    x = torch.randn(4, 7, 32)
    lo, hi = -8, 7
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / hi
    codes = torch.clamp(torch.round(x / scale), lo, hi)
    q._act_codes, q._act_scale, q._exact_path = codes, scale, True
    with torch.no_grad():
        y = q(x)
    wcodes, gamma = q._codes_and_gamma()
    ref = (codes.double() * scale.double()) @ (wcodes.double() * gamma.double()).transpose(0, 1)
    rel = ((y.double() - ref).abs() / ref.abs().clamp(min=1e-9)).max()
    assert rel < 1e-3, f"code-matmul relative error {rel:.2e}"
    # exactness: small integer terms accumulate exactly in fp32
    exact = (codes.long().float() @ wcodes.long().float().transpose(0, 1)) * scale * gamma
    assert torch.equal(y, exact), "fp32 accumulation must be exact for these magnitudes"


# ---------- code cache (the budget-critical T5 change) ----------

def test_code_cache_invalidates_on_weight_mutation():
    torch.manual_seed(3)
    q = QLinear(8, 8)
    q.weight.data.normal_()
    c1, _ = q._codes_and_gamma()
    _, _ = q._codes_and_gamma()
    assert q._codes is c1, "codes must be cached across forwards"
    with torch.no_grad():
        q.weight.add_(0.1)  # in-place update -> tensor._version bump -> invalidate
    c3, _ = q._codes_and_gamma()
    assert c3 is not c1, "cache must invalidate after an optimizer-style update"
    w = q.weight.detach()
    gamma = w.abs().mean()
    ref = torch.clamp(torch.round(w / (gamma + 1e-5)), -1, 1)
    assert torch.equal(c3, ref)


def test_ste_grad_flows_through_cached_codes():
    torch.manual_seed(4)
    q = QLinear(8, 4)
    ref = torch.nn.Linear(8, 4, bias=False)
    ref.weight.data.copy_(q.weight.data)
    x = torch.randn(2, 8)
    q(x).sum().backward()
    ref(x).sum().backward()
    assert torch.allclose(q.weight.grad, ref.weight.grad), "identity weight STE must hold"
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()


def test_latent_must_be_fp32():
    with pytest.raises(ValueError):
        QLinear(4, 4, dtype=torch.bfloat16)
    src = torch.nn.Linear(4, 4).to(torch.bfloat16)
    q = swap_linear(src, "ternary")
    assert q.weight.dtype == torch.float32 and q.weight.device == src.weight.device


def test_exact_path_is_dust_only():
    torch.manual_seed(5)
    q = QLinear(8, 4)
    q.weight.data.normal_()
    x = torch.randn(2, 3, 8)
    lo, hi = -8, 7
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / hi
    garbage = torch.clamp(torch.round((x + 5.0) / scale), lo, hi)  # WRONG codes
    w_eff = q.effective_weight()
    import torch.nn.functional as F
    ref = F.linear(x, w_eff)  # no hooks in this unit test: the module receives RAW x
    q._act_codes, q._act_scale = garbage, scale
    with torch.no_grad():
        # no hooks here: the module receives RAW x, so the BP reference is F.linear(raw x)
        assert torch.allclose(q(x), ref), "BP semantics: stash must be IGNORED (fake-quant)"
    # the fake-quant path clears the stash (stale-value hygiene) — re-arm it
    q._act_codes, q._act_scale = garbage, scale
    q._exact_path = True
    with torch.no_grad():
        assert not torch.allclose(q(x), ref), "DUST mode: stash must be USED"


def test_dust_forward_config_plumbs(tmp_path):
    """OQ-40 plumbing: dust_forward reaches QLinear._exact_mode on the Dust path; "fake"
    must make the estimator share BP-STE's quantized forward (Plan v3.1 4.2)."""
    from dustq.config import RunConfig
    from dustq.train import Runner
    for mode in ("tf32", "fake"):
        cfg = RunConfig(task="smoke", size="tiny", tokens="1M", precision="w1.58a4",
                        method="dust", seed=0, data_dir="synthetic", steps=2,
                        eval_interval=2, dust_forward=mode, output_dir=str(tmp_path)).resolved()
        r = Runner(cfg)
        r._build()
        modes = {m._exact_mode for m in r.model.modules()
                 if isinstance(m, QLinear)}
        assert modes == {mode}, f"dust_forward={mode} did not reach the QLinear modules"
        r.model = None  # release the tiny model before the next iteration

def test_smoke_bp_sste(tmp_path):
    from dustq.config import RunConfig
    from dustq.train import Runner
    cfg = RunConfig(task="smoke", size="tiny", tokens="1M", precision="w1.58a4",
                    method="backprop", seed=0, data_dir="synthetic", steps=3,
                    eval_interval=2, lr=1e-3, momentum=0.9, ste_mode="smoothed",
                    output_dir=str(tmp_path)).resolved()
    r = Runner(cfg)
    r.train()
    result_path = tmp_path / cfg.run_id / "result.json"
    import json
    history = json.loads(result_path.read_text())["history"]
    losses = [rec["train_loss"] for rec in history if "train_loss" in rec]
    assert len(losses) == 3 and all(math.isfinite(l) for l in losses)


def test_sste_forward_unchanged_and_smoothed_slope_bounded():
    """The adaptive width and the Delta factor must not move the forward value (SSTE and STE
    share it), must stay finite, and must keep dQ/dx in the range the smoothed quantizer can
    produce (its average slope is 1 at any width; the A4 grid's clamp keeps it near 1)."""
    torch.manual_seed(0)
    x = torch.tensor([0.0, 0.03, -0.07, 0.21], requires_grad=True)
    scale = torch.full_like(x, 0.4)
    for sigma in (0.05, 0.2, 0.4, 1.6, 6.4):
        if x.grad is not None:
            x.grad = None
        y = _SSTEActQuant.apply(x, scale.clone(), -8, 7, sigma)
        assert torch.equal(y, torch.clamp(torch.round(x.detach() / 0.4), -8, 7) * 0.4), \
            "the forward must be the hard-quantized value at every sigma"
        y.sum().backward()
        g = x.grad.clone()
        assert torch.isfinite(g).all() and (g > 0).all(), (sigma, g)
        assert g.mean() < 3.0, f"sigma={sigma}: mean slope {g.mean():.3f} is above the 1/Delta=2.5 ceiling"




# ---------- 2026-11-08 audit regressions ----------

def test_group_of_covers_value_embedding_sites():
    """ve_gate and value_embeds used to fall through to their raw parameter names, which
    (a) made any script filtering on canonical groups silently drop them and (b) changed the
    group count, hence the median reliability and every disattenuated number."""
    from dustq.probes import SITE_GROUPS, group_of

    for name, expected in [
        ("transformer.h.0.attn.ve_gate.weight", SITE_GROUPS["attn.ve_gate"]),
        ("transformer.h.3.attn.ve_gate.weight", SITE_GROUPS["attn.ve_gate"]),
        ("value_embeds.1.weight", SITE_GROUPS["value_embeds"]),
        ("value_embeds.3.weight", SITE_GROUPS["value_embeds"]),
        # the eight original groups must be unchanged
        ("transformer.h.0.attn.c_q.weight", "Q"),
        ("transformer.h.1.attn.c_proj.weight", "Attn out"),
        ("transformer.h.0.mlp.c_fc.weight", "MLP in"),
        ("transformer.h.2.mlp.c_proj.weight", "MLP out"),
        ("transformer.wte.weight", "Tok emb"),
        ("lm_head.weight", "Head"),
    ]:
        assert group_of(name) == expected, (name, group_of(name), expected)


def test_mse_alpha_scale_is_floored():
    """mse_alpha had no clamp (absmax and the weight path both clamp at 1e-5), so a token with
    all-zero activations produced scale = 0 and 0/0 = NaN - and in the SSTE backward,
    sigma_u = inf, an unbounded boundary loop."""
    import math

    from dustq.quant import ste_act_quantize

    x = torch.zeros(4, 16)
    y = ste_act_quantize(x, 4, scheme="mse_alpha", alpha=0.5)
    assert torch.isfinite(y).all(), f"mse_alpha produced non-finite output: {y}"
    # a floor also means a tiny-but-nonzero activation still quantizes to a finite code
    x2 = torch.full((4, 16), 1e-12)
    y2 = ste_act_quantize(x2, 4, scheme="mse_alpha", alpha=0.5)
    assert torch.isfinite(y2).all(), f"tiny activations produced non-finite output: {y2}"


def test_tf32_int_matmul_restores_previous_flag():
    """_tf32_int_matmuls used to set allow_tf32 = False unconditionally in `finally`, which on
    Ampere/Ada/Hopper (where PyTorch defaults it to True) permanently disabled TF32 for the
    rest of the process - a silent global slowdown. See docs/REMEDIATION-2026-11-08.md R8."""
    import dustq.quant as quant_mod

    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        a = torch.randint(-8, 8, (8, 8)).float()
        b = torch.randint(-8, 8, (8, 8)).float()
        for value in (True, False):
            torch.backends.cuda.matmul.allow_tf32 = value
            out = quant_mod._tf32_int_matmuls(a, b)
            assert torch.backends.cuda.matmul.allow_tf32 == value, (
                f"the helper clobbered allow_tf32 (was {value}, now "
                f"{torch.backends.cuda.matmul.allow_tf32})")
            assert torch.equal(out, a @ b)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior
