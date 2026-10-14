"""T1 tests: config resolution, exact resume (the acceptance criterion), queue idempotence.

Resume acceptance (Plan v3.1 section 7): a run stopped mid-way and resumed must match an
uninterrupted run exactly — same noise stream (DUSTQ generator state), same optimizer
momentum, same data schedule — hence bitwise-identical final weights.
"""

import json
import math
import subprocess
import sys
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
    """Review-2: bitwise on CPU (the acceptance evidence); tolerance on CUDA, where SDPA
    backward and bf16 ops are not guaranteed run-to-run deterministic."""
    exact = next(m1.parameters()).device.type == "cpu"
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and v1.dtype == v2.dtype, f"keys/dtypes differ at {k1}"
        if exact:
            assert torch.equal(v1, v2), f"weights differ at {k1}"
        else:
            assert torch.allclose(v1.float(), v2.float(), rtol=1e-2, atol=1e-2), f"weights differ at {k1}"


def assert_close(a, b, device):
    if device.type == "cpu":
        assert a == b
    else:
        assert math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-3)


def assert_histories_match(h1, h2, device):
    assert len(h1) == len(h2)
    for r1, r2 in zip(h1, h2):
        assert r1["step"] == r2["step"]
        for key in ("train_loss", "val_loss"):
            if key in r1 or key in r2:
                assert (key in r1) == (key in r2), f"step {r1['step']} mismatch on {key}"
                assert_close(r1[key], r2[key], device)


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

    dev = next(run_a.model.parameters()).device
    assert_histories_match(strip_timing(result_a["history"]), strip_timing(result_c["history"]), dev)
    assert_close(result_a["best_val"], result_c["best_val"], dev)
    assert_close(result_a["test_at_best_val"], result_c["test_at_best_val"], dev)
    assert_same_state(run_a.model, run_c.model)
    # M2: no duplicate step rows in metrics.jsonl after the resume
    steps_seen = [json.loads(line)["step"] for line in
                  (tmp_path / "b" / cfg_c.run_id / "metrics.jsonl").read_text().splitlines()
                  if line.strip()]
    assert steps_seen == sorted(set(steps_seen)) and steps_seen == [1, 2, 3, 4]


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

    dev = next(run_a.model.parameters()).device
    assert_histories_match(strip_timing(result_a["history"]), strip_timing(result_c["history"]), dev)
    assert_close(result_a["test_at_best_val"], result_c["test_at_best_val"], dev)
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


# --- Review-fix tests (docs/specs/2026-10-11-review-fixes-design.md) ---

def test_dust_lr_override_reaches_optimizer(tmp_path):
    # B5: tuned lr/momentum must reach the Dust optimizer, not just the runs.csv log.
    cfg = tiny_cfg(lr=0.001, momentum=0.5, output_dir=str(tmp_path))
    r = Runner(cfg)
    r._build()
    base = [g for g in r.optimizer.param_groups if g["lr"] not in (1000.0, 0.3, 0.03)]
    assert base and base[0]["lr"] == pytest.approx(0.001)
    assert base[0]["momentum"] == pytest.approx(0.5)


@pytest.mark.skipif(not Path("data/train_1M.pt").exists(), reason="data files not prepared")
def test_truncated_run_uses_full_budget_schedule(tmp_path):
    # C2: a truncated run must see exactly the full-budget run's batches (upstream --steps).
    from dust import load_sequences
    cfg = RunConfig(task="smoke", size="L4d256", tokens="1M", method="backprop", seed=0,
                    data_dir="data", steps=2, output_dir=str(tmp_path)).resolved()
    r = Runner(cfg)
    r._build()
    r._load_data()
    train = load_sequences("data/train_1M.pt", 61 * 8)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed + 1))
    assert torch.equal(r.training, train[order])


def test_resume_does_not_corrupt_best(tmp_path):
    # C3: after a crash with a stale ckpt, the resumed run must not overwrite best.pt
    # with worse weights; the file's best value is authoritative.
    from dustq.train import load_best
    cfg_b = tiny_cfg(output_dir=str(tmp_path / "b"), steps=2)
    Runner(cfg_b).train()
    run_dir = tmp_path / "b" / cfg_b.run_id
    (run_dir / "result.json").unlink()  # a real kill leaves no result.json
    best, state = load_best(run_dir / "best.pt", "cpu")
    crafted = {"best": 7.0, "state": state}  # a "best" no later tiny-run eval can beat
    torch.save(crafted, run_dir / "best.pt.tmp")
    (run_dir / "best.pt.tmp").replace(run_dir / "best.pt")
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    ckpt["best"] = 99.0  # stale checkpoint, as after a crash before the last ckpt save
    torch.save(ckpt, run_dir / "ckpt.pt")

    cfg_c = tiny_cfg(output_dir=str(tmp_path / "b"), steps=4)
    Runner(cfg_c).train()
    best_after, state_after = load_best(run_dir / "best.pt", "cpu")
    assert best_after == 7.0
    assert torch.equal(state_after["lm_head.weight"], state["lm_head.weight"])
    result = json.loads((run_dir / "result.json").read_text())
    assert result["best_val"] == 7.0


def test_queue_failure_marker(tmp_path):
    # M1: deterministic failures retry at most twice, then skip; --retry-failed resets.
    import yaml
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg = tiny_cfg(method="backprop", lr=1e-3, momentum=0.9,
                   data_dir=str(tmp_path / "missing"), steps=1)
    (configs / "bad.yaml").write_text(yaml.safe_dump(cfg.to_yaml_dict()))
    results = str(tmp_path / "results")

    assert all(s.startswith("FAILED") for s in run_queue(configs, results))
    assert all(s.startswith("FAILED") for s in run_queue(configs, results))
    assert all(s.startswith("skip") for s in run_queue(configs, results))
    assert all(s.startswith("FAILED") for s in run_queue(configs, results, retry_failed=True))

    lines = (tmp_path / "results" / "runs.csv").read_text().splitlines()
    assert len(lines) == 4  # header + 3 failed rows (the skip adds none)
    assert sum(1 for line in lines if ",failed," in line) == 3


def test_benchmark_end_to_end_tiny():
    # B2/B3 class guard: the benchmark script must run end to end (aliases + spellings),
    # incl. the quantized cells, and write its report.
    r = subprocess.run([sys.executable, "scripts/benchmark.py", "--device", "cpu", "--scale", "tiny",
                        "--sizes", "S", "--populations", "256", "--precisions", "fp,w1.58a4",
                        "--steps", "1", "--skip-concurrent"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert Path("results/benchmark_cpu.md").exists()
    assert "s/step" in r.stdout


def test_benchmark_worker_mode():
    # M3: the subprocess worker mode used by the 2-concurrent measurement.
    r = subprocess.run([sys.executable, "scripts/benchmark.py", "--device", "cpu", "--scale", "tiny",
                        "--worker", "--worker-size", "L4d256", "--worker-population", "256",
                        "--worker-precision", "fp", "--worker-steps", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert "s_per_step" in r.stdout
