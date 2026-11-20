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
