"""T1 tests: config resolution, exact resume (the acceptance criterion), queue idempotence.

Resume acceptance (Plan v3.1 section 7): a run stopped mid-way and resumed must match an
uninterrupted run exactly — same noise stream (DUSTQ generator state), same optimizer
momentum, same data schedule — hence bitwise-identical final weights.
"""

import json
from pathlib import Path

import pytest
import torch

from dustq.config import RunConfig
from dustq.queue import run_queue
from dustq.train import Runner, evaluate


def tiny_cfg(**over):
    base = dict(task="smoke", size="tiny", tokens="1M", precision="fp", method="dust",
                population=256, seed=0, data_dir="synthetic", steps=4, eval_interval=2,
                checkpoint_minutes=0, notes="t1-test")
    base.update(over)
    return RunConfig(**base).resolved()


def assert_same_state(m1, m2):
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and v1.dtype == v2.dtype and torch.equal(v1, v2), f"weights differ at {k1}"


def strip_timing(history):
    return [{k: v for k, v in rec.items() if k != "elapsed_seconds"} for rec in history]


def test_config_resolution_and_run_id():
    d = RunConfig(size="L4d256", tokens="1M", method="dust", population=256).resolved()
    assert (d.lr, d.momentum) == (0.25, 0.95) and d.steps == 61 and d.eval_interval == 2
    assert d.run_id == "L4d256_1M_fp_dust_K256_s0"
    b = RunConfig(size="L4d256", tokens="1M", method="backprop").resolved()
    assert (b.lr, b.momentum) == (0.175, 0.98) and b.run_id == "L4d256_1M_fp_backprop_K0_s0"
    t = RunConfig(size="L8d512", tokens="10M", method="dust", population=16384).resolved()
    assert (t.lr, t.momentum) == (0.25, 0.95)  # upstream 10M/16k cell
    q = RunConfig(size="L4d256", tokens="1M", precision="w1.58a4", method="dust",
                  population=1024, seed=2).resolved()
    assert q.run_id == "L4d256_1M_w1p58a4_dust_K1024_s2"


def test_resume_exactness_dust(tmp_path):
    cfg_a = tiny_cfg(output_dir=str(tmp_path / "a"))
    run_a = Runner(cfg_a)
    run_a.train()
    result_a = json.loads((tmp_path / "a" / cfg_a.run_id / "result.json").read_text())

    cfg_b = tiny_cfg(output_dir=str(tmp_path / "b"), steps=2)      # "killed" after step 2
    Runner(cfg_b).train()
    (tmp_path / "b" / cfg_b.run_id / "result.json").unlink()      # a real kill leaves no result.json
    cfg_c = tiny_cfg(output_dir=str(tmp_path / "b"), steps=4)      # resume from step 2
    run_c = Runner(cfg_c)
    run_c.train()
    result_c = json.loads((tmp_path / "b" / cfg_c.run_id / "result.json").read_text())

    assert strip_timing(result_a["history"]) == strip_timing(result_c["history"])
    assert result_a["best_val"] == result_c["best_val"]
    assert result_a["test_at_best_val"] == result_c["test_at_best_val"]
    assert_same_state(run_a.model, run_c.model)


def test_resume_exactness_backprop(tmp_path):
    bp = dict(method="backprop", precision="w1.58a4", lr=1e-3, momentum=0.9)
    cfg_a = tiny_cfg(output_dir=str(tmp_path / "a"), **bp)
    run_a = Runner(cfg_a)
    run_a.train()
    result_a = json.loads((tmp_path / "a" / cfg_a.run_id / "result.json").read_text())

    cfg_b = tiny_cfg(output_dir=str(tmp_path / "b"), steps=2, **bp)
    Runner(cfg_b).train()
    (tmp_path / "b" / cfg_b.run_id / "result.json").unlink()
    cfg_c = tiny_cfg(output_dir=str(tmp_path / "b"), steps=4, **bp)
    run_c = Runner(cfg_c)
    run_c.train()
    result_c = json.loads((tmp_path / "b" / cfg_c.run_id / "result.json").read_text())

    assert strip_timing(result_a["history"]) == strip_timing(result_c["history"])
    assert result_a["test_at_best_val"] == result_c["test_at_best_val"]
    assert_same_state(run_a.model, run_c.model)


def test_queue_idempotent(tmp_path):
    import yaml
    configs = tmp_path / "configs"
    configs.mkdir()
    for name, over in [("a_dust.yaml", dict(method="dust", steps=2)),
                       ("b_bp.yaml", dict(method="backprop", precision="w1.58a4",
                                          steps=2, lr=1e-3, momentum=0.9))]:
        cfg = tiny_cfg(**over)
        (configs / name).write_text(yaml.safe_dump(cfg.to_yaml_dict()))
    results = str(tmp_path / "results")

    statuses = run_queue(configs, results)
    assert all(s.startswith(("done", "FAILED")) for s in statuses), statuses
    statuses2 = run_queue(configs, results)
    assert all(s.startswith("skip") for s in statuses2), statuses2

    csv_text = (tmp_path / "results" / "runs.csv").read_text().splitlines()
    assert len(csv_text) == 3  # header + 2 runs
    ids = [line.split(",")[0] for line in csv_text[1:]]
    assert len(set(ids)) == 2


def test_evaluate_matches_upstream_semantics(tmp_path):
    from dustq.runner import build_model
    model = build_model(precision="fp", seed=0)
    rows = torch.randint(0, 4096, (4, 129))
    loss = evaluate(model, rows, torch.device("cpu"))
    assert 7.5 < loss < 9.5  # random tokens against a fresh model: near ln(4096) = 8.3
