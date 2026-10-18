"""Idempotent run queue (Plan v3.1 section 4.4): every session runs `scripts/queue.py`.

For each config: skip completed runs (result.json exists), resume partial ones (ckpt.pt),
start fresh otherwise. Every completed or failed run is appended to results/runs.csv
(the v1 section 6 schema). Failed runs are logged, never deleted.
"""

from __future__ import annotations

import csv
import gc
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
        tokens=cfg.tokens, precision=cfg.precision, method=cfg.method,
        K=cfg.population if cfg.method == "dust" else 0,  # v1 convention: backprop rows are K0
        seed=cfg.seed, lr=cfg.lr, momentum=cfg.momentum,
        noise_scales=json.dumps(cfg.sigmas) if cfg.sigmas else "default",
        val_loss_best=result.get("best_val", ""), test_loss_at_best_val=result.get("test_at_best_val", ""),
        steps=cfg.steps, wall_s=result.get("wall_s", ""), gpu=result.get("gpu", ""),
        dtype=result.get("dtype", ""), cu_burn_rate=cfg.cu_burn_rate,
        commit=result.get("commit", ""), status=status, notes=notes,
    )


FAILED_MARKER = "FAILED_ATTEMPTS"
MAX_ATTEMPTS = 2


def _attempts(run_dir: Path) -> int:
    path = run_dir / FAILED_MARKER
    if not path.exists():
        return 0
    try:
        return int(path.read_text().splitlines()[0] or 0)
    except (ValueError, IndexError):
        return 0


def run_queue(configs_dir, results_dir="results", only: str | None = None,
              cu_burn_rate: str = "", retry_failed: bool = False) -> list[str]:
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
        if retry_failed and (run_dir / FAILED_MARKER).exists():
            (run_dir / FAILED_MARKER).unlink()  # explicit override; rows are never deleted
        if _attempts(run_dir) >= MAX_ATTEMPTS:
            statuses.append(f"skip {cfg.run_id} (failed {MAX_ATTEMPTS}x)")
            continue
        cfg.output_dir = str(results_dir)  # runs land under the queue's results dir
        failed = False
        try:
            runner = Runner(cfg, device=None)
            result_path = runner.train()
            result = json.loads(result_path.read_text())
            _append_csv_row(csv_path, _row_from(cfg, result, "completed"))
            marker = run_dir / FAILED_MARKER
            if marker.exists():  # a success clears the failure history for this run
                marker.unlink()
            statuses.append(f"done {cfg.run_id}")
            done.append(cfg.run_id)
        except Exception as exc:  # log the failure, keep the queue moving
            note = f"{type(exc).__name__}: {exc}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / FAILED_MARKER).write_text(f"{_attempts(run_dir) + 1}\n{note}\n")
            # Full stack for the next debugging session; keep the tail (the failing frames
            # are at the end). The 500-char CSV note truncated exactly the frames that
            # identified the eval-grads OOM and cost a day of wrong root-causing.
            (run_dir / "FAILED_TRACEBACK.txt").write_text(
                "".join(traceback.format_exception(exc))[-20000:])
            _append_csv_row(csv_path, _row_from(cfg, {}, "failed", notes=note[:500]))
            traceback.print_exc()
            statuses.append(f"FAILED {cfg.run_id}: {note}")
            failed = True
        if failed:
            # Outside the except block (exception released): a failed run's model/optimizer
            # and any zombie autograd graphs are reference cycles — force collection so the
            # next config starts from clean memory.
            gc.collect()
    return statuses
