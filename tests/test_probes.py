
"""T6 probe tests: run against a tiny backprop checkpoint, CPU, synthetic-free (train 3
steps of backprop fp on the tiny model via the Runner, save to a tmp dir)."""

import json
from pathlib import Path

import pytest
import torch

from dustq.config import RunConfig
from dustq.probes import (ProbeContext, cosine_by_group, fit_cos_law, probe_cosine_ladders,
                          probe_directional_descent, probe_reference, probe_sigma_delta,
                          probe_ste_bias)


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

