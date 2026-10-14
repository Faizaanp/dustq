"""Phase 0 acceptance tests (Plan v3.1 section 7): recipes, bitwise DUSTQ == DUST, smoke steps.

The bitwise test is the contract that DUSTQ.step is upstream's step with sigma literals
replaced by config: identical fp models + identical seeds must produce identical noise and
therefore bitwise-identical gradients.
"""

import math

import pytest
import torch

from dustq import qmodel, recipes, runner
from dustq.config import SIZES
from dustq.dust_ext import DUSTQ
from dustq.quant import swap_linear, ste_act_quantize
from dustq.upstream import ensure_upstream_on_path

ensure_upstream_on_path()

import dust  # noqa: E402

K_VALUES = (256, 1024, 4096, 16384)


def test_l8_recipes_byte_equal():
    for tokens in ("1M", "10M"):
        for population in K_VALUES:
            assert recipes.recipe(tokens, population, 8) == dust.recipe(tokens, population), \
                f"divergence at tokens={tokens} K={population}"


def test_l4_l2_merged_recipes():
    r4 = recipes.recipe("1M", 256, 4)
    assert r4.writers == [70, 26, 10, 4]
    assert r4.hubs == [12, 6, 4, 4]
    assert sum(r4.writers) + sum(r4.hidden) + sum(r4.hubs) + r4.embedding == 256
    r2 = recipes.recipe("1M", 256, 2)
    assert r2.writers == [96, 14]
    assert r2.hubs == [18, 8]
    for tokens in ("1M", "10M"):
        for population in K_VALUES:
            for n_layer in (2, 4):
                r = recipes.recipe(tokens, population, n_layer)
                counts = list(r.writers) + list(r.hidden) + list(r.hubs) + [r.embedding] + list(r.local.values())
                assert all(c % r.nrep == 0 for c in counts), \
                    f"indivisible counts at tokens={tokens} K={population} L={n_layer}"


def test_default_sigmas_match_upstream_literals():
    from dustq.dust_ext import DEFAULT_SIGMAS
    assert DEFAULT_SIGMAS == {
        "writer": 0.2, "hidden": 0.2, "emb": 0.2, "hub_shallow": 0.2, "hub_deep": 0.4,
        "head": 0.05, "local": 0.05, "scalar": 0.03,
    }


def test_dustq_bitwise_equals_dust():
    model_a = runner.build_model(precision="fp", seed=0)
    model_b = runner.build_model(precision="fp", seed=0)
    for (_, pa), (_, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.equal(pa, pb), "same-seed make_model must produce identical weights"

    settings = recipes.recipe("1M", 256, model_a.config.n_layer)
    est_a = dust.DUST(model_a, settings, seed=0)
    est_b = DUSTQ(model_b, settings, seed=0)
    opt_a = dust.make_optimizer(model_a, settings)
    opt_b = dust.make_optimizer(model_b, settings)
    x, y = runner.tiny_batch(torch.device("cpu"))

    loss_a = est_a.step(x, y, opt_a)
    loss_b = est_b.step(x, y, opt_b)
    assert loss_a == loss_b, "same noise stream must give the same clean loss"

    for (name_a, pa), (name_b, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        assert pa.grad is not None and pb.grad is not None
        assert torch.equal(pa.grad, pb.grad), f"gradient mismatch at {name_a}"
    est_a.close()
    est_b.close()


@pytest.mark.parametrize("precision", ["fp", "w1.58a4"])
def test_smoke_dustq(precision):
    losses = runner.run_dust(precision=precision, steps=3)
    assert len(losses) == 3
    assert all(math.isfinite(l) for l in losses)


def test_smoke_bp_ste():
    losses = runner.run_backprop(precision="w1.58a4", steps=3)
    assert len(losses) == 3
    assert all(math.isfinite(l) for l in losses)


def test_quantizer_semantics():
    from dustq.quant import QLinear, ste_act_quantize
    q = QLinear(64, 32, mode="ternary")
    q.weight.data.normal_()
    eff = q.effective_weight().detach()
    gamma = q.weight.abs().mean()
    unit = gamma.detach().unique()
    values = (eff / gamma.detach()).round().unique()
    assert torch.all(values.abs() <= 1 + 1e-6), "ternary effective weight must be gamma * {-1,0,1}"

    x = torch.randn(2, 8, 64)
    xq = ste_act_quantize(x, 4, unsigned=False)
    # Reconstruct codes with the quantizer's own per-token scale (absmax / hi, hi = 7 for signed A4).
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-5) / 7
    codes = (xq / scale).round().unique()
    assert torch.all(codes >= -8 - 1e-6) and torch.all(codes <= 7 + 1e-6), \
        "signed A4 codes must lie in [-8, 7]"
    xu = torch.nn.functional.relu(x).square()
    xqu = ste_act_quantize(xu, 4, unsigned=True)
    assert float(xqu.min()) >= -1e-6, "unsigned grid must be non-negative"

    # STE identity: gradient flows through unchanged.
    x2 = torch.randn(4, 16, requires_grad=True)
    ste_act_quantize(x2, 8).sum().backward()
    assert torch.all(x2.grad == 1), "identity STE must pass gradient 1 everywhere"

# --- Review-fix tests (docs/specs/2026-10-11-review-fixes-design.md) ---

@pytest.mark.parametrize("size", sorted(SIZES))
def test_size_forwards(size):
    # B1: every experiment size must build and forward (n_kv_head now matches n_head).
    model = runner.build_model(SIZES[size], "fp", "cpu", seed=0)
    assert model.config.n_kv_head == model.config.n_head
    out = model(torch.randint(0, 4096, (2, 64)))
    assert out.shape == (2, 64, 4096) and torch.isfinite(out).all()


def test_precision_spellings():
    # B3: dotted and run-id spellings both accepted.
    assert qmodel.parse_precision("w1.58a4") == ("ternary", 4, "absmax")
    assert qmodel.parse_precision("w1p58a4") == ("ternary", 4, "absmax")
    assert qmodel.parse_precision("W1_58_A2") == ("ternary", 2, "absmax")


def test_swap_preserves_dtype_and_device():
    # B4: the QLinear swap must be transparent to device and dtype.
    src = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    q = swap_linear(src, "ternary")
    assert q.weight.dtype == src.weight.dtype and q.weight.device == src.weight.device
    assert q.effective_weight().dtype == src.weight.dtype


def test_act_quantizer_fp32_math():
    # C1: quantizer math runs in fp32 regardless of input dtype.
    xb = (torch.randn(64, 128) * 3).bfloat16()
    out = ste_act_quantize(xb, 8)
    assert out.dtype == torch.float32
    assert torch.equal(out, ste_act_quantize(xb.float(), 8))


def test_lowbit_absmax_guard():
    # C5: bits <= 4 with absmax is refused outside smoke/benchmark mode.
    model = runner.build_model(precision="w1.58a4", seed=0)
    with pytest.raises(ValueError):
        qmodel.register_act_hooks(model, 4, "absmax")
    handles = qmodel.register_act_hooks(model, 4, "absmax", allow_lowbit_absmax=True)
    for h in handles:
        h.remove()
