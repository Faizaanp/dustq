"""Lightweight CPU tests for config validation and checkpoint/JSONL resume edge cases (C1-C6)."""

import json
import pytest
import torch

from dustq.config import RunConfig, load_config
from dustq.train import Runner


def test_config_validation_edges(tmp_path):
    """C5 & C6: empty YAML and unsupported tokens/population raise clear ValueError."""
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_config(empty_yaml)

    with pytest.raises(ValueError, match="Unsupported tokens budget"):
        RunConfig(tokens="20M", method="dust").resolved()

    with pytest.raises(ValueError, match="Unsupported Dust population"):
        RunConfig(tokens="1M", method="dust", population=999).resolved()


def test_resume_legacy_ckpt_and_truncated_metrics(tmp_path):
    """C1, C2, C3: resume survives legacy ckpt missing optimizer, best=None, and truncated JSONL."""
    cfg = RunConfig(
        task="smoke", size="tiny", tokens="1M", precision="fp", method="backprop",
        seed=0, data_dir="synthetic", steps=2, eval_interval=1, output_dir=str(tmp_path),
    ).resolved()
    runner = Runner(cfg, device="cpu")
    runner._build()
    runner._load_data()

    # Legacy checkpoint missing "optimizer" and having best=None
    torch.save({
        "model": runner.model.state_dict(),
        "step": 1,
        "best": None,
        "history": [],
    }, runner.ckpt_path)

    # Corrupted trailing line in metrics.jsonl (mid-write disconnect simulation)
    metrics_path = runner.run_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 1, "train_loss": 8.0, "val_loss": 8.0}) + "\n"
        + '{"step": 2, "train_lo'  # truncated
    )

    res_path = runner.train()
    res = json.loads(res_path.read_text())
    assert res["steps"] == 2
    assert res["best_val"] is not None and res["best_val"] > 0
    lines = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    assert [l["step"] for l in lines] == [1, 2]


def test_load_best_handles_none_best(tmp_path):
    """F7 regression: load_best must not crash with TypeError: float(None)."""
    from dustq.train import load_best
    p = tmp_path / "best.pt"
    torch.save({"best": None, "state": {"w": torch.ones(2)}}, p)
    val, state = load_best(p, "cpu")
    assert val == 0.0 and "w" in state


def test_probe_schema_version_gate(tmp_path):
    """F2 / O2 regression: stale pre-fix probe 2 & 4 JSONs without schema_version >= 2
    must be rejected by is_probe_report_current, while v2 JSONs and unchanged probe 5 pass."""
    from dustq.probes import PROBE_SCHEMA_VERSION, is_probe_report_current

    # Pre-fix probe 2 JSON (missing schema_version) -> must NOT be accepted as done
    stale_p2 = tmp_path / "stale_p2.json"
    stale_p2.write_text(json.dumps({
        "cosine_ladders": {"256": {}},
        "ste_bias": {},
        "disattenuation": {"divisor": "sqrt(spearman_brown(r_g))"},
    }))
    assert is_probe_report_current(stale_p2, "3,2") is False

    # Post-fix v2 probe 2 JSON -> accepted
    fresh_p2 = tmp_path / "fresh_p2.json"
    fresh_p2.write_text(json.dumps({
        "schema_version": PROBE_SCHEMA_VERSION,
        "cosine_ladders": {"256": {}},
        "ste_bias": {},
        "disattenuation": {"rung_seeds": {"256": 10000}},
    }))
    assert is_probe_report_current(fresh_p2, "3,2") is True

    # Pre-fix probe 4 JSON (missing schema_version and directional_descent_seeds) -> rejected
    stale_p4 = tmp_path / "stale_p4.json"
    stale_p4.write_text(json.dumps({"directional_descent": {"Dust1k": {}}}))
    assert is_probe_report_current(stale_p4, "4") is False

    # Post-fix v2 probe 4 JSON -> accepted
    fresh_p4 = tmp_path / "fresh_p4.json"
    fresh_p4.write_text(json.dumps({
        "schema_version": PROBE_SCHEMA_VERSION,
        "directional_descent": {"Dust1k": {}},
        "directional_descent_seeds": {"Dust1k": {"n_seeds": 3}},
    }))
    assert is_probe_report_current(fresh_p4, "4") is True

    # Unchanged probe 5 JSON from v1 -> still accepted without unnecessary rerun
    p5_v1 = tmp_path / "p5_v1.json"
    p5_v1.write_text(json.dumps({"sigma_delta": {"cos": {}}}))
    assert is_probe_report_current(p5_v1, "5") is True
