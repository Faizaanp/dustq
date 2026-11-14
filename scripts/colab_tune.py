"""Dust-Q Task T7 Hyperparameter Tuning session on Colab L4 — paste into ONE cell and run.

Plan v3.1 section 6.5 & AGENTS.md:
- Pre-requisite: A fresh dustq.zip uploaded to Colab (/content/dustq.zip).
- Evaluates LR and momentum grids across P0-P4 on S/1M validation loss using seeds 100+.
- Strictly selection by validation loss; test loss is never used for selection.
- Outputs results/tune_best_hparams.json containing winning hyperparameters for Tier B.
- Fully idempotent: can be stopped and resumed at any time without duplicating work.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ZIP = "/content/dustq.zip"
PY = sys.executable
T0 = time.monotonic()

# ---- 0. Environment check ----
import torch

assert torch.cuda.is_available(), "Restart with an L4 GPU runtime"
print("gpu:", torch.cuda.get_device_name(0), flush=True)

import zipfile

if not Path(REPO_ZIP).is_file():
    if Path("dustq.zip").is_file():
        REPO_ZIP = "dustq.zip"
    else:
        raise SystemExit(f"MISSING {REPO_ZIP}! Please upload dustq.zip to Colab (/content/dustq.zip) before running.")

with zipfile.ZipFile(REPO_ZIP) as z:
    z.extractall("/content/dustq-repo")
base = Path("/content/dustq-repo")
root = next(p for p in [base] + [q for q in base.iterdir() if q.is_dir()] if (p / "dustq").is_dir())
os.chdir(root)
print("repo at:", root, flush=True)

# ---- 1. Mount Google Drive and set symlinks ----
if not (Path("/content/drive/MyDrive").is_dir() or Path("/content/drive/My Drive").is_dir()):
    from google.colab import drive
    drive.mount("/content/drive")

drive_candidates = [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]
drive_root = next((p for p in drive_candidates if p.is_dir()), None)
if drive_root is None:
    raise SystemExit("Neither /content/drive/MyDrive nor /content/drive/My Drive found!")

DRIVE_DATA = drive_root / "dustq" / "data"
DRIVE_RESULTS = drive_root / "dustq" / "results"
DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)

for link_name, target in [("data", DRIVE_DATA), ("results", DRIVE_RESULTS)]:
    p = Path(link_name)
    if os.path.lexists(link_name):
        if p.is_symlink() and p.resolve() == target.resolve() and p.exists():
            continue
        p.unlink(missing_ok=True)
    if not p.exists():
        os.symlink(target, link_name)

def run_checked(cmd, **kwargs):
    """Stream stdout/stderr line-by-line so Colab never appears frozen."""
    print("RUNNING:", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )
    lines = []
    if proc.stdout is not None:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
            lines.append(line)
            if len(lines) > 2000:
                lines.pop(0)
        proc.stdout.close()
    returncode = proc.wait()
    if returncode != 0:
        tail = "".join(lines[-100:])
        print(f"\nCOMMAND FAILED (exit {returncode}): {' '.join(str(c) for c in cmd)}", flush=True)
        print("--- Output Tail ---\n" + tail, flush=True)
        raise SystemExit(f"exit {returncode}; see output above")
    return subprocess.CompletedProcess(cmd, returncode, stdout="".join(lines), stderr="")


# ---- 2. Preflight checks ----
print("\n=== Preflight checks (data, alpha table) ===", flush=True)
for need, label in ((Path("data/train_1M.pt"), "data/train_1M.pt"),
                    (Path("data/val.pt"), "data/val.pt"),
                    (Path("configs/alpha_calibrated.json"), "alpha_calibrated.json")):
    if not need.is_file():
        raise SystemExit(f"MISSING {label} — please check Google Drive data and configs.")

print("Preflight checks passed.", flush=True)

# ---- 3. Ensure Tuning Configs are emitted ----
print("\n=== Generating Task T7 Tuning Configs ===", flush=True)
run_checked([PY, "scripts/make_configs.py", "--tier", "tune", "--out", "configs"])

tune_configs = list(Path("configs").glob("*_tune_*.yaml"))
print(f"Total tuning configs ready: {len(tune_configs)}", flush=True)
assert len(tune_configs) == 114, f"Expected 114 tuning configs, got {len(tune_configs)}"

# ---- 4. Run Tuning Queue ----
print("\n=== Running Task T7 Tuning Queue (S/1M, seeds 100+) ===", flush=True)
# Allow overriding ONLY filter via environment variable (e.g. for chunked runs)
filter_substr = os.environ.get("TUNE_ONLY", "_tune")
run_checked([PY, "scripts/run_queue.py", "--only", filter_substr, "--cu-burn-rate", "1.94"])

# ---- 5. Post-run Analysis & Summary Table ----
print("\n=== Analyzing Task T7 Results & Selecting Best Hyperparameters ===", flush=True)
run_checked([PY, "scripts/analyze_tuning.py", "--results-dir", "results", "--out", "results/tune_best_hparams.json"])

elapsed = time.monotonic() - T0
cu_spent = (elapsed / 3600.0) * 1.94
print(f"\nTuning session completed in {elapsed/3600.0:.2f} h (~{cu_spent:.2f} CU spent).", flush=True)
print("Winning hyperparameters saved to results/tune_best_hparams.json on Drive.", flush=True)
