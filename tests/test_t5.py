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

def test_sste_backward_matches_mc():
    torch.manual_seed(0)
    lo, hi, scale, sigma = -8, 7, 0.05, 0.05
    x = torch.tensor([0.0, 0.025, 0.02, 0.05, 0.30, -0.025, -0.10], requires_grad=True)
    y = _SSTEActQuant.apply(x, torch.full_like(x, scale), lo, hi, sigma)
    y.sum().backward()
    gx = x.grad.clone()

    q = lambda z: torch.clamp(torch.round(z / scale), lo, hi)
    x0 = x.detach()
    n, h = 400_000, 0.002
    eps = torch.randn(n, x0.numel())
    mc = (q(x0 + h + sigma * eps).mean(0) - q(x0 - h + sigma * eps).mean(0)) / (2 * h)
    assert torch.allclose(gx, mc, atol=0.25), (gx, mc)

    # shape sanity at small sigma/grid ratio (sigma_u = 0.2): flat mid-step, peaked boundary
    xs = torch.tensor([0.0, 0.025], requires_grad=True)  # mid-step vs boundary (scale 0.05)
    ys = _SSTEActQuant.apply(xs, torch.full_like(xs, 0.05), lo, hi, 0.01)
    ys.sum().backward()
    assert xs.grad[0] < 0.3 * xs.grad[1], "mid-step must be flatter than the boundary peak"


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
