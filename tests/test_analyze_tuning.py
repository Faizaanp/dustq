"""Unit tests for scripts/analyze_tuning.py."""

import csv
from pathlib import Path
import pytest

from scripts.analyze_tuning import analyze_tuning, print_summary_table


def test_analyze_tuning_selection(tmp_path):
    csv_path = tmp_path / "runs.csv"

    # Create dummy runs with various val and test losses
    rows = [
        # Cell 1: w1.58a4 backprop smoothed
        # Run A: val 5.10, test 5.20
        # Run B: val 5.05, test 5.30 (lower val loss, higher test loss -> MUST WIN)
        # Run C: val 5.15, test 4.90 (lower test loss -> MUST LOSE)
        {
            "run_id": "L4d256_1M_w1p58a4_backprop_K0_s100_sste_tune_lr0.175_m0.9",
            "task": "tune",
            "precision": "w1.58a4",
            "method": "backprop",
            "ste_mode": "smoothed",
            "K": "0",
            "lr": "0.175",
            "momentum": "0.9",
            "val_loss_best": "5.10",
            "test_loss_at_best_val": "5.20",
            "status": "completed",
        },
        {
            "run_id": "L4d256_1M_w1p58a4_backprop_K0_s100_sste_tune_lr0.13125_m0.98",
            "task": "tune",
            "precision": "w1.58a4",
            "method": "backprop",
            "ste_mode": "smoothed",
            "K": "0",
            "lr": "0.13125",
            "momentum": "0.98",
            "val_loss_best": "5.05",
            "test_loss_at_best_val": "5.30",
            "status": "completed",
        },
        {
            "run_id": "L4d256_1M_w1p58a4_backprop_K0_s100_sste_tune_lr0.2625_m0.9",
            "task": "tune",
            "precision": "w1.58a4",
            "method": "backprop",
            "ste_mode": "smoothed",
            "K": "0",
            "lr": "0.2625",
            "momentum": "0.9",
            "val_loss_best": "5.15",
            "test_loss_at_best_val": "4.90",
            "status": "completed",
        },
        # Cell 2: w1.58a4 dust K256
        {
            "run_id": "L4d256_1M_w1p58a4_dust_K256_s100_fake_tune_lr0.25_m0.95",
            "task": "tune",
            "precision": "w1.58a4",
            "method": "dust",
            "ste_mode": "",
            "K": "256",
            "lr": "0.25",
            "momentum": "0.95",
            "val_loss_best": "5.32",
            "test_loss_at_best_val": "5.31",
            "status": "completed",
        },
        {
            "run_id": "L4d256_1M_w1p58a4_dust_K256_s100_fake_tune_lr0.375_m0.95",
            "task": "tune",
            "precision": "w1.58a4",
            "method": "dust",
            "ste_mode": "",
            "K": "256",
            "lr": "0.375",
            "momentum": "0.95",
            "val_loss_best": "5.28",
            "test_loss_at_best_val": "5.29",
            "status": "completed",
        },
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = analyze_tuning(csv_path)

    assert "w1.58a4/backprop/smoothed" in best
    bp_win = best["w1.58a4/backprop/smoothed"]
    # Run B must win because val_loss_best 5.05 < 5.10 < 5.15
    assert bp_win["best_lr"] == 0.13125
    assert bp_win["best_momentum"] == 0.98
    assert bp_win["best_val_loss"] == 5.05
    assert bp_win["test_loss_at_best_val"] == 5.30
    assert bp_win["grid_evaluated"] == 3

    assert "w1.58a4/dust/K256" in best
    dust_win = best["w1.58a4/dust/K256"]
    assert dust_win["best_lr"] == 0.375
    assert dust_win["best_val_loss"] == 5.28
    assert dust_win["grid_evaluated"] == 2

    # Verify table printer executes without error
    print_summary_table(best)
