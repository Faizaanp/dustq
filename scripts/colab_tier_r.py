"""Dust-Q Tier R launch on Colab L4 — paste this whole file into ONE cell and run.

Runs, in order: GPU check -> repo unzip -> Drive mount + data check -> the pending
2-concurrent measurement (3 min) -> the Tier R queue with RESULTS AND CHECKPOINTS ON
DRIVE, so any disconnect resumes exactly where it left off.

Before running (one-time setup):
  1. Runtime -> Change runtime type -> L4 GPU.
  2. Upload the fresh `dustq.zip` (repo zip from HEAD) via the Files panel.
  3. Upload the data files to Google Drive at MyDrive/dustq/data/ — train_1M.pt,
     train_10M.pt, val.pt, test.pt, manifest.json (~50 MB; one-time, reused forever).

Expected: Tier R is ~37 L4-h (~56 CU at 1.52 CU/h) — plan on 3-4 sessions; the queue is
idempotent, so re-running this cell in a new session just continues.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ZIP = "/content/dustq.zip"
DRIVE_DATA = "/content/drive/MyDrive/dustq/data"
DRIVE_RESULTS = "/content/drive/MyDrive/dustq/results"
CU_BURN_RATE = "1.52"  # measured 2026-10-16; update from the Resources tab if it differs

# ---- 0. environment ----
import torch
assert torch.cuda.is_available(), "Restart with an L4 GPU runtime"
print("gpu:", torch.cuda.get_device_name(0),
      "| vram:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")

# ---- 1. repo ----
import zipfile
with zipfile.ZipFile(REPO_ZIP) as z:
    z.extractall("/content/dustq-repo")
base = Path("/content/dustq-repo")
root = next(p for p in [base] + [q for q in base.iterdir() if q.is_dir()] if (p / "dustq").is_dir())
os.chdir(root)
print("repo at:", root)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True)

# ---- 2. drive + data ----
from google.colab import drive
drive.mount("/content/drive")
for name in ("train_1M.pt", "train_10M.pt", "val.pt", "test.pt"):
    assert (Path(DRIVE_DATA) / name).is_file(), (
        f"missing {name} — upload the data files to Drive at MyDrive/dustq/data/")
if not Path("data").exists():
    os.symlink(DRIVE_DATA, "data")  # configs use data_dir: data
Path(DRIVE_RESULTS).mkdir(parents=True, exist_ok=True)
print("data ok (Drive); results ->", DRIVE_RESULTS)

# ---- 3. quick GPU sanity (6 s) ----
subprocess.run([sys.executable, "-m", "pytest", "tests/test_cuda.py", "-q"], check=True)

# ---- 4. the pending 2-concurrent measurement (3 min; feeds the CUT_CELLS buy-backs) ----
t0 = time.time()
procs = [subprocess.Popen([sys.executable, "scripts/benchmark.py", "--device", "cuda",
                           "--worker-scale", "real", "--worker-size", "L4d256",
                           "--worker-population", "256", "--worker-precision", "fp",
                           "--worker-steps", "2"]) for _ in range(2)]
codes = [p.wait() for p in procs]
wall = time.time() - t0
serial = 4 * 10.141  # 2 jobs x 2 steps at the measured single-job rate
print(f"returncodes: {codes}")
print(f"2-concurrent S/K256: {wall:.1f} s vs {serial:.1f} s serial -> {serial / wall:.2f}x")
print("^ PASTE THIS LINE BACK (buy-back decision, docs/CUT_CELLS.md)")

# ---- 5. Tier R queue (idempotent; resumable; results on Drive) ----
cmd = [sys.executable, "scripts/run_queue.py", "--configs-dir", "configs",
       "--results-dir", DRIVE_RESULTS, "--cu-burn-rate", CU_BURN_RATE]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print("\nTier R queue finished for this session. Re-run this cell any time; it resumes.")
print("When all 18 configs show done/skip, paste runs.csv + the 3 seed means for the G0 check.")
