"""Idempotent run queue (Plan v3.1 section 4.4): every session runs `scripts/queue.py`.

For each config: skip completed runs (result.json exists), resume partial ones (ckpt.pt),
start fresh otherwise. Every completed or failed run is appended to results/runs.csv
(the v1 section 6 schema). Failed runs are logged, never deleted.
"""

from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path

from .config import CSV_COLUMNS, RunConfig, load_configs_dir
from .train import Runner


def _append_csv_row(csv_path: Path, row: dict):
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _row_from(cfg: RunConfig, result: dict, status: str, notes: str = "") -> dict:
    return dict(
        run_id=cfg.run_id, task=cfg.task, model=cfg.size, params=result.get("params", ""),
        tokens=cfg.tokens, precision=cfg.precision, method=cfg.method, K=cfg.population,
        seed=cfg.seed, lr=cfg.lr, momentum=cfg.momentum,
        noise_scales=json.dumps(cfg.sigmas) if cfg.sigmas else "default",
        val_loss_best=result.get("best_val", ""), test_loss_at_best_val=result.get("test_at_best_val", ""),
        steps=cfg.steps, wall_s=result.get("wall_s", ""), gpu=result.get("gpu", ""),
        dtype=result.get("dtype", ""), cu_burn_rate=cfg.cu_burn_rate,
        commit=result.get("commit", ""), status=status, notes=notes,
    )


def run_queue(configs_dir, results_dir="results", only: str | None = None,
              cu_burn_rate: str = "") -> list[str]:
    configs_dir, results_dir = Path(configs_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "runs.csv"
    done, statuses = [], []
    for path, cfg in load_configs_dir(configs_dir):
        if only and only not in cfg.run_id:
            continue
        if cu_burn_rate:
            cfg.cu_burn_rate = cu_burn_rate
        run_dir = results_dir / cfg.run_id
        if (run_dir / "result.json").exists():
            statuses.append(f"skip {cfg.run_id} (complete)")
            continue
        cfg.output_dir = str(results_dir)  # runs land under the queue's results dir
        try:
            runner = Runner(cfg, device=None)
            result_path = runner.train()
            result = json.loads(result_path.read_text())
            _append_csv_row(csv_path, _row_from(cfg, result, "completed"))
            statuses.append(f"done {cfg.run_id}")
            done.append(cfg.run_id)
        except Exception as exc:  # log the failure, keep the queue moving
            note = f"{type(exc).__name__}: {exc}"
            _append_csv_row(csv_path, _row_from(cfg, {}, "failed", notes=note[:500]))
            traceback.print_exc()
            statuses.append(f"FAILED {cfg.run_id}: {note}")
    return statuses
