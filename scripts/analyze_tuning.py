"""Analyze Task T7 hyperparameter tuning results from results/runs.csv.

Selection protocol (Plan v3.1 section 6.5; AGENTS.md Rule 2):
- Strictly by validation loss (val_loss_best). No test-set tuning.
- Tuning seeds are 100+ and never count as reported experimental seeds.
- Outputs the winning LR and momentum per (precision, method, ste_mode/K) cell.
- Emits results/tune_best_hparams.json for use by Tier B config generation.

Usage:
  python scripts/analyze_tuning.py [--results-dir results] [--out results/tune_best_hparams.json]
"""

import argparse
import csv
import json
from pathlib import Path


def parse_float(val: str | float | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def analyze_tuning(csv_path: Path) -> dict:
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} does not exist.")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to completed tuning runs
    tune_rows = [
        r for r in rows
        if (r.get("task") == "tune" or "_tune" in r.get("run_id", ""))
        and r.get("status") == "completed"
        and parse_float(r.get("val_loss_best")) is not None
    ]

    if not tune_rows:
        return {}

    # Group by cell:
    # - backprop: (precision, "backprop", ste_mode)
    # - dust: (precision, "dust", K)
    groups = {}
    for r in tune_rows:
        prec = r.get("precision")
        method = r.get("method")
        if method == "backprop":
            mode = r.get("ste_mode", "identity")
            cell_key = f"{prec}/backprop/{mode}"
        elif method == "dust":
            k = int(r.get("K") or 0)
            cell_key = f"{prec}/dust/K{k}"
        else:
            continue

        groups.setdefault(cell_key, []).append(r)

    best_hparams = {}
    for cell_key, items in groups.items():
        # Sort strictly by val_loss_best ascending (lower is better)
        sorted_items = sorted(items, key=lambda x: float(x["val_loss_best"]))
        best = sorted_items[0]

        best_hparams[cell_key] = {
            "cell": cell_key,
            "precision": best.get("precision"),
            "method": best.get("method"),
            "ste_mode": best.get("ste_mode"),
            "K": int(best.get("K") or 0),
            "best_lr": float(best.get("lr")),
            "best_momentum": float(best.get("momentum")),
            "best_val_loss": float(best.get("val_loss_best")),
            "test_loss_at_best_val": parse_float(best.get("test_loss_at_best_val")),
            "best_run_id": best.get("run_id"),
            "grid_evaluated": len(items),
            "all_runs": [
                {
                    "run_id": x.get("run_id"),
                    "lr": float(x.get("lr")),
                    "momentum": float(x.get("momentum")),
                    "val_loss": float(x.get("val_loss_best")),
                    "test_loss": parse_float(x.get("test_loss_at_best_val")),
                }
                for x in sorted_items
            ],
        }

    return best_hparams


def print_summary_table(best_hparams: dict):
    print("\n" + "=" * 90)
    print("TASK T7 TUNING RESULTS — WINNING HYPERPARAMETERS (BY VALIDATION LOSS ONLY)")
    print("=" * 90)
    print(f"{'Cell':<26} | {'Grid':<5} | {'Best LR':<10} | {'Mom':<5} | {'Val Loss':<10} | {'Test Loss':<10}")
    print("-" * 90)

    # Sort cells logically: fp, w8a8, w1.58a8, w1.58a4, w1.58a2
    prec_order = ["fp", "w8a8", "w1p58a8", "w1.58a8", "w1p58a4", "w1.58a4", "w1p58a2", "w1.58a2"]
    def sort_key(k):
        prec = k.split("/")[0]
        p_idx = prec_order.index(prec) if prec in prec_order else 99
        return (p_idx, k)

    for cell in sorted(best_hparams.keys(), key=sort_key):
        info = best_hparams[cell]
        v_loss = f"{info['best_val_loss']:.5f}" if info['best_val_loss'] is not None else "N/A"
        t_loss = f"{info['test_loss_at_best_val']:.5f}" if info['test_loss_at_best_val'] is not None else "N/A"
        print(f"{cell:<26} | {info['grid_evaluated']:<5} | {info['best_lr']:<10.6g} | {info['best_momentum']:<5.2f} | {v_loss:<10} | {t_loss:<10}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Directory containing runs.csv")
    parser.add_argument("--out", default=None, help="Output JSON path (default: results-dir/tune_best_hparams.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "runs.csv"
    out_path = Path(args.out) if args.out else results_dir / "tune_best_hparams.json"

    if not csv_path.is_file():
        print(f"Error: {csv_path} not found.")
        return 1

    best_hparams = analyze_tuning(csv_path)
    if not best_hparams:
        print(f"No completed tuning runs found in {csv_path}.")
        return 0

    print_summary_table(best_hparams)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_hparams, indent=2))
    print(f"Summary written to {out_path}")
    return 0


if __name__ == "__main__":
    main()
