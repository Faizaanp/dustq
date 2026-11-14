"""Tests for T7 tuning and Tier B config generators (Plan v3.1 sections 6.2, 6.5, 7)."""

import pytest
import yaml

from dustq.config import RunConfig, load_config
from scripts.make_configs import P3_SIGMAS, tier_a, tier_b_m, tier_b_s, tier_r, tier_tune


def test_tier_tune_isolation_and_sigmas():
    """T7 rules: seeds >= 100, no collisions, P3 has optimal sigmas from Probe 5."""
    cfgs = tier_tune(seeds=[100])
    assert len(cfgs) == 114

    seen_ids = set()
    for c in cfgs:
        resolved = c.resolved()
        # Seed isolation rule
        assert resolved.seed >= 100, f"Tuning seed must be >= 100, got {resolved.seed}"
        assert resolved.task == "tune"
        # Unique run_id rule
        assert resolved.run_id not in seen_ids, f"Collision detected: {resolved.run_id}"
        seen_ids.add(resolved.run_id)

        # Check P3 (w1.58a4) optimal sigmas
        if resolved.precision == "w1.58a4":
            if resolved.method == "dust" or (resolved.method == "backprop" and resolved.ste_mode == "smoothed"):
                assert resolved.sigmas == P3_SIGMAS

        # Suffix checks
        assert "_tune" in resolved.run_id
        assert "lr" in resolved.run_id
        assert "_m" in resolved.run_id


def test_tier_b_disambiguation():
    """Tier B rules: task='B' produces _tierB suffix, distinct seeds."""
    bs_cfgs = tier_b_s(seeds=[0, 1, 2])
    seen_ids = set()
    for c in bs_cfgs:
        resolved = c.resolved()
        assert "_tierB" in resolved.run_id
        assert resolved.run_id not in seen_ids
        seen_ids.add(resolved.run_id)

    bm_cfgs = tier_b_m(seeds=[0, 1, 2])
    for c in bm_cfgs:
        resolved = c.resolved()
        assert "_tierB" in resolved.run_id
        assert resolved.run_id not in seen_ids
        seen_ids.add(resolved.run_id)


def test_config_yaml_roundtrip(tmp_path):
    """Generated configs must roundtrip via load_config without loss of parameters."""
    cfg = tier_tune(seeds=[100])[0].resolved()
    path = tmp_path / f"{cfg.run_id}.yaml"
    path.write_text(yaml.safe_dump(cfg.to_yaml_dict(), sort_keys=False))

    loaded = load_config(path).resolved()
    assert loaded.run_id == cfg.run_id
    assert loaded.lr == cfg.lr
    assert loaded.momentum == cfg.momentum
    assert loaded.task == cfg.task
    assert loaded.precision == cfg.precision


def test_tier_b_with_tuned_hparams():
    """Tier B config generator must apply tuned hyperparameters when provided."""
    dummy_tuned = {
        "w1.58a4/backprop/smoothed": {
            "best_lr": 0.13125,
            "best_momentum": 0.9,
            "best_val_loss": 5.05,
        },
        "w1.58a4/dust/K256": {
            "best_lr": 0.375,
            "best_momentum": 0.95,
            "best_val_loss": 5.28,
        },
    }
    cfgs = tier_b_s(seeds=[0], tuned=dummy_tuned)
    for c in cfgs:
        resolved = c.resolved()
        if resolved.precision == "w1.58a4":
            if resolved.method == "backprop" and resolved.ste_mode == "smoothed":
                assert resolved.lr == 0.13125
                assert resolved.momentum == 0.9
            elif resolved.method == "dust" and resolved.population == 256:
                assert resolved.lr == 0.375
                assert resolved.momentum == 0.95

