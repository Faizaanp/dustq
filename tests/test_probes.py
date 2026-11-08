
"""T6 probe tests: run against a tiny backprop checkpoint, CPU, synthetic-free (train 3
steps of backprop fp on the tiny model via the Runner, save to a tmp dir)."""

import json
from pathlib import Path

import pytest
import torch

from dustq.config import RunConfig
from dustq.probes import (ProbeContext, cosine_by_group, fit_cos_law, probe_cosine_ladders,
                          probe_directional_descent, probe_reference, probe_seed_noise,
                          probe_sigma_delta, probe_ste_bias)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("probes")
    cfg2 = RunConfig(task="smoke", size="tiny", tokens="1M", precision="w1.58a4",
                     method="dust", seed=0, data_dir="synthetic", steps=3, eval_interval=2,
                     output_dir=str(tmp)).resolved()
    Runner = __import__("dustq.train", fromlist=["Runner"]).Runner
    Runner(cfg2).train()  # 3 tiny steps -> checkpoint on disk at tmp/cfg2.run_id/best.pt
    ctx_obj = ProbeContext.__new__(ProbeContext)  # skip __post_init__ data loading
    from dustq.backprop_ext import make_bp_optimizer
    from dustq.runner import build_model, tiny_batch
    from dustq.train import load_best
    ckpt = tmp / cfg2.run_id / "best.pt"
    _, state = load_best(ckpt, torch.device("cpu"))
    ctx_obj.cfg = cfg2
    ctx_obj.device = torch.device("cpu")
    ctx_obj.model = build_model(cfg2.model_kwargs(), "w1.58a4", "cpu", seed=0).requires_grad_(True)
    ctx_obj.model.load_state_dict(state)
    ctx_obj.optimizer = make_bp_optimizer(ctx_obj.model, 1e-3, 0.9)
    ctx_obj.x, ctx_obj.y = tiny_batch(torch.device("cpu"), sequences=8, length=128)
    ctx_obj.vx, ctx_obj.vy = tiny_batch(torch.device("cpu"), seed=99, sequences=8, length=128)
    ctx_obj.batch = (ctx_obj.x, ctx_obj.y)
    ctx_obj.heldout = (ctx_obj.vx, ctx_obj.vy)
    ctx_obj.best_val = 0.0
    yield ctx_obj


def test_collect_and_cosine_helpers(ctx):
    g = ctx.dust_grads(seed=0, population=256)
    h = ctx.dust_grads(seed=1000, population=256)
    assert set(g) and set(h)
    cg = cosine_by_group(g, h)
    assert all(-1.001 <= v <= 1.001 for v in cg.values())
    assert "Attn out" in cg and "MLP in" in cg


def test_reference_and_ladders(ctx):
    _, ref, rel = probe_reference(ctx)
    assert -1.0 <= rel <= 1.0
    lad = probe_cosine_ladders(ctx, populations=(256, 1024), reference=ref)
    assert set(lad) == {256, 1024}
    # reporting precedent (arXiv:2609.19589): raw and disattenuated both present
    for population, groups in lad.items():
        for name, val in groups.items():
            assert set(val) == {"raw", "disattenuated"}
            assert val["disattenuated"] >= val["raw"] - 1e-6


def test_ste_bias_probe(ctx):
    _, ref, _ = probe_reference(ctx)
    bias = probe_ste_bias(ctx, reference=ref)
    assert set(bias["ste"]) == set(bias["sste"])
    assert all(-1.001 <= v <= 1.001 for v in bias["ste"].values())


def test_sigma_delta_probe(ctx):
    _, ref, _ = probe_reference(ctx)
    sweep = probe_sigma_delta(ctx, ref, sigma_fractions=(1, 2), populations=(256,))
    assert len(sweep) == 2


def test_directional_descent_probe(ctx):
    cand = {"Dust256": ctx.dust_grads(seed=0, population=256)}
    dd = probe_directional_descent(ctx, cand)
    assert "train" in dd["Dust256"] and "val" in dd["Dust256"]


def test_cos_law_fit_shape():
    cos = {"MLP out": {64: 0.2, 256: 0.5, 1024: 0.7, 4096: 0.8}, "Head": {64: 0.1, 256: 0.3}}
    fits = fit_cos_law(cos)
    assert fits["MLP out"] is not None and fits["MLP out"]["c_max"] > 0.8
    assert fits["Head"] is None, "fewer than 3 K-points must return None"



def test_seed_noise_probe(ctx):
    """Probe 0: seed-pair cosines on the same checkpoint/batch are the floor any
    within-seed-noise claim (incl. G1's +0.9910) must be benchmarked against."""
    noise = probe_seed_noise(ctx, populations=(256,))
    assert set(noise) == {"256"}
    for groups in noise.values():
        assert groups and all(-1.001 <= v <= 1.001 for v in groups.values())


def test_probe_context_sets_dust_forward(ctx):
    """OQ-40 in the probes: ProbeContext plumbs cfg.dust_forward into every QLinear so
    Dust and backprop probes share one quantized forward (Plan v3.1 4.2)."""
    obj = ProbeContext(ctx.cfg, batch=ctx.batch, heldout=ctx.heldout)
    assert obj.x.shape == ctx.x.shape and torch.equal(obj.x, ctx.x)
    modes = {m._exact_mode for m in obj.model.modules() if hasattr(m, "_exact_mode")}
    assert modes == {ctx.cfg.dust_forward}, f"dust_forward did not reach the QLinear modules: {modes}"
    obj._ensure_hooks("identity")
    assert obj._hook_handles, "act hooks must register for a quantized probe model"


def test_pckpt_task_probe_path(tmp_path):
    """Regression for the 2026-11-02 Colab crash: the real run uses task="PCKPT"
    (no smoke override), so w1.58a4 resolves through mse_alpha + the FROZEN alpha table
    and hooks register with allow_lowbit_absmax=False. That combination never ran on CPU
    before (the 6/6 suite always used task=smoke -> absmax). Tiny sizes keep it CPU-fast;
    seq length is the L4 size but only 2 steps are trained."""
    cfg = RunConfig(task="PCKPT", size="tiny", tokens="1M", precision="w1.58a4",
                    method="backprop", seed=0, data_dir="synthetic", steps=2,
                    eval_interval=2, output_dir=str(tmp_path)).resolved()
    assert cfg.task == "PCKPT" and cfg.dust_forward == "tf32"  # config default, not smoke
    Runner = __import__("dustq.train", fromlist=["Runner"]).Runner
    Runner(cfg).train()  # 2 quantized backprop steps -> best.pt with mse_alpha path

    from dustq.runner import tiny_batch
    ctx_obj = ProbeContext(cfg, batch=tiny_batch(torch.device("cpu"), sequences=4),
                           heldout=tiny_batch(torch.device("cpu"), seed=99, sequences=4))
    # the exact failing invocation's path, minus the expensive reference probe
    ctx_obj._ensure_hooks("identity")
    assert ctx_obj._hook_handles, "act hooks must register on the real mse_alpha path"
    g_ste = ctx_obj.bp_grads("identity")
    g_sste = ctx_obj.bp_grads("smoothed")
    assert set(g_ste) == set(g_sste) and g_ste
    g_dust = ctx_obj.dust_grads(seed=0, population=256)
    groups = cosine_by_group(g_dust, g_ste)
    assert groups and all(-1.001 <= v <= 1.001 for v in groups.values())
    # gamma of the quantized forward must actually have run: per-token scales exist
    assert all(p.isfinite().all() for p in g_dust.values())


def test_site_sigma_key_mapping():
    """qmodel.site_sigma_key is the inverse map probe 5 needs: every act-quant site must
    resolve to a real DUSTQ sigma key (the old code passed SITE names like 'attn.c_q'
    straight to DUSTQ, which raises 'unknown sigma keys')."""
    from dustq import qmodel
    from dustq.dust_ext import DEFAULT_SIGMAS

    names = ["transformer.h.0.attn.c_q", "transformer.h.0.attn.c_k",
             "transformer.h.1.attn.c_v", "transformer.h.0.attn.c_proj",
             "transformer.h.3.attn.c_proj", "transformer.h.0.mlp.c_fc",
             "transformer.h.1.mlp.c_proj"]
    for name in names:
        key = qmodel.site_sigma_key(name, 4)
        assert key in DEFAULT_SIGMAS, f"{name} -> {key!r} is not a DUSTQ sigma key"
    # the hub sigma splits by depth: shallow half 0.2, deep half 0.4 (n_layer=4)
    assert qmodel.site_sigma("transformer.h.0.attn.c_proj", 4) == 0.2
    assert qmodel.site_sigma("transformer.h.3.attn.c_proj", 4) == 0.4
    assert qmodel.site_sigma("transformer.h.1.attn.c_q", 4) == 0.05
    assert qmodel.site_sigma("transformer.h.0.mlp.c_fc", 4) == 0.2
    assert qmodel.site_sigma("transformer.h.0.mlp.c_proj", 4) == 0.2
    # sites without an act hook fall back, and are excluded from the sweep keys
    assert qmodel.site_sigma_key("transformer.wte", 4) is None
    assert qmodel.site_sigma_key("lm_head", 4) is None


def _hook_qualnames(module):
    """Ordered qualnames of a module's forward pre-hooks, to identify whose hook is whose."""
    return [getattr(fn, "__qualname__", "?") for fn in module._forward_pre_hooks.values()]


def test_dust_grads_hook_order(ctx):
    """The estimator's o@l noise hook must come BEFORE our act-quant pre-hooks (noise ->
    quantize), on EVERY dust_grads call. The probes registered the act hooks first, which
    both reversed the order and - on the exact paths - dropped the jitter entirely, since
    QLinear's codes are computed from the input the act hook sees. DUSTQ.close() removes
    Dust's hooks but leaves ours in place, so the ordering must be re-established per call."""
    from dustq.dust_ext import DUSTQ

    def qualnames(module_list):
        return _hook_qualnames(module_list)

    seen = []
    original_step = DUSTQ.step

    def recording_step(self, x, y, opt):
        # at step time the module's pre-hook order is whatever dust_grads() established
        seen.append(qualnames(ctx.model.get_submodule("transformer.h.0.attn.c_proj")))
        return original_step(self, x, y, opt)

    DUSTQ.step = recording_step
    try:
        ctx.dust_grads(seed=0, population=256)
        ctx.dust_grads(seed=1, population=256)
    finally:
        DUSTQ.step = original_step

    assert len(seen) == 2, "both dust_grads calls should have recorded"
    for i, hooks in enumerate(seen):
        dust = next(j for j, q in enumerate(hooks) if q.startswith("DUST.input_hook"))
        ours = next(j for j, q in enumerate(hooks) if q.startswith("register_act_hooks"))
        assert dust < ours, (f"call {i}: the o@l noise hook (index {dust}) must run BEFORE "
                             f"the act-quant hook (index {ours}): noise -> quantize")


def test_sigma_delta_sweep(ctx):
    """Probe 5: the sweep must actually vary sigma (the old build-and-ignore bug), key it by
    real DUSTQ sigma keys, and report a flip rate in [0, 1]."""
    from dustq.probes import probe_sigma_delta
    from dustq.dust_ext import DEFAULT_SIGMAS

    # any per-parameter gradient dict works as the reference; a backprop one keeps this
    # test cheap (probe 5 only needs something to take cosines against)
    ref = ctx.bp_grads("identity")
    sweep = probe_sigma_delta(ctx, ref, sigma_fractions=(1.0, 4.0), populations=(256,))
    assert set(sweep) == {(1.0, 256), (4.0, 256)}
    for (frac, _), payload in sweep.items():
        assert set(payload["sigma"]) <= set(DEFAULT_SIGMAS), payload["sigma"]
        rates = payload["flip_rate"]
        assert rates and all(0.0 <= v <= 1.0 for v in rates.values()), rates
        cos = payload["cos"]
        assert cos and all(-1.001 <= v <= 1.001 for v in cos.values())
    # sigma really moved between the two points, and by exactly the frac ratio
    # (payload sigmas are rounded to 6 decimals, so compare at that resolution)
    lo, hi = sweep[(1.0, 256)]["sigma"], sweep[(4.0, 256)]["sigma"]
    for key in lo:
        assert abs(hi[key] - 4.0 * lo[key]) < 1e-5, (key, lo[key], hi[key])
    # the jitter is not a no-op: at least one group's cosine changes with sigma
    xs = [sweep[(f, 256)]["cos"] for f in (1.0, 4.0)]
    assert any(abs(a[g] - b[g]) > 1e-6 for a, b in zip(xs, xs[1:]) for g in a), \
        "the sweep produced identical cosines at 1x and 4x sigma - sigmas were not applied"
    # a larger jitter must move at least as many codes as a smaller one
    assert sweep[(4.0, 256)]["flip_rate"] != sweep[(1.0, 256)]["flip_rate"]


def test_probes_cli_main_end_to_end(tmp_path, monkeypatch):
    """Regression for the 2026-11-02 reporting crash. No test ever exercised main() - the
    fixture tests call the probes directly - so a KeyError in the sigma-Delta table reached
    the L4 after the full sweep had already computed. Runs the real CLI path in-process with
    the expensive K=16k reference swapped for a cheap one (the reference's QUALITY is not
    what this test is about; its plumbing and every reporting block are)."""
    import sys

    import yaml

    import dustq.probes as probes

    cfg = RunConfig(task="PCKPT", size="tiny", tokens="1M", precision="w1.58a4",
                    method="backprop", seed=0, data_dir="synthetic", steps=2,
                    eval_interval=2, output_dir=str(tmp_path)).resolved()
    Runner = __import__("dustq.train", fromlist=["Runner"]).Runner
    Runner(cfg).train()
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg.to_yaml_dict()), encoding="utf-8")

    cheap_ref = probes.ProbeContext(cfg).bp_grads("identity")
    monkeypatch.setattr(probes, "probe_reference", lambda *a, **k: (None, cheap_ref, 0.99))
    monkeypatch.setattr(sys, "argv", [
        "dustq.probes", "--config", str(yaml_path),
        "--probes", "3,2,5,0", "--out", str(tmp_path / "probes")])
    probes.main()

    path = tmp_path / "probes" / f"{cfg.run_id}_probes_0-2-3-5.json"
    assert path.is_file(), f"main() must write its report JSON at {path}"
    report = json.loads(path.read_text())
    assert {"seed_noise", "reliability", "cosine_ladders", "ste_bias", "sigma_delta"} <= set(report)
    assert report["reliability"] == 0.99
    sd = report["sigma_delta"]
    assert sd["cos"] and sd["flip_rate"] and sd["sigma"], \
        "the sweep must report cos, flip rate and sigma (the crash here lost all of it)"
    assert all(-1.001 <= v <= 1.001 for groups in sd["cos"].values() for v in groups.values())
    for groups in sd["flip_rate"].values():
        assert all(0.0 <= v <= 1.0 for v in groups.values()), groups


def test_ladders_use_the_reliability_they_are_given(ctx):
    """Regression for the 2026-11-07 fix: the CLI computed a reference reliability (0.29-0.38
    on the real checkpoints) and then called probe_cosine_ladders WITHOUT passing it, so the
    function defaulted to 1.0 and every shipped "disattenuated" value equalled the raw cosine
    while the same JSON reported the real reliability. Fast by design - a backprop reference,
    one population, and a synthetic reliability - so it needs no K=4096 estimator steps."""
    ctx_local = ProbeContext(ctx.cfg, batch=(ctx.x, ctx.y), heldout=(ctx.vx, ctx.vy))
    ref = ctx_local.bp_grads("identity")
    lad = probe_cosine_ladders(ctx_local, populations=(256,), reference=ref, reliability=0.5)
    assert set(lad) == {256}
    for groups in lad.values():
        assert groups, "no groups reported"
        for name, val in groups.items():
            expected = max(-1.0, min(1.0, val["raw"] / 0.5))
            assert abs(val["disattenuated"] - expected) < 1e-12, (name, val)
            assert -1.0 <= val["disattenuated"] <= 1.0, (name, val)

    # and the default must stay 1.0 when no reliability is supplied (unchanged behaviour)
    lad_default = probe_cosine_ladders(ctx_local, populations=(256,), reference=ref)
    for groups in lad_default.values():
        for val in groups.values():
            assert val["disattenuated"] == val["raw"], (
                "omitting reliability must keep the old 1.0 default, not invent one")
