"""Dust-Q T6 probe session on Colab L4 — paste into ONE cell and run.

Requires: fresh dustq.zip; L4 runtime; Drive data at MyDrive/dustq/data (present).
Runs, in order (Plan v3.1 section 5):
  1. The STE/SSTE-bias probe (probe 3) FIRST per plan — the cheap H2 gate. If cos(STE, ref)
     at P3 is already at the fp level, H2 is unlikely and Tier B re-decides before spending.
  2. Probes 2 + 1 (cosine ladders, disattenuated) and 5 (sigma-Delta sweep) per precision.
  3. Probe 4 (directional descent) on S at P3 (the primary bit width).
  4. An M spot-check at P3 (ladders + a 3-point sweep) only if the L4 fraction is used.
Outputs: results/probes/<run_id>_probes.json plus printed tables; copy the JSON back.
Expected ~2.5-3.5 h, ~5 CU. The residual alpha table is regenerated if missing.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ZIP = "/content/dustq.zip"
DRIVE_DATA = "/content/drive/MyDrive/dustq/data"
DRIVE_RESULTS = "/content/drive/MyDrive/dustq/results"
PY = sys.executable
T0 = time.monotonic()

# ---- 0. environment ----
import torch

assert torch.cuda.is_available(), "Restart with an L4 GPU runtime"
print("gpu:", torch.cuda.get_device_name(0))

import zipfile

with zipfile.ZipFile(REPO_ZIP) as z:
    z.extractall("/content/dustq-repo")
base = Path("/content/dustq-repo")
root = next(p for p in [base] + [q for q in base.iterdir() if q.is_dir()] if (p / "dustq").is_dir())
os.chdir(root)
print("repo at:", root)

from google.colab import drive

drive.mount("/content/drive")
if not Path("data").exists():
    os.symlink(DRIVE_DATA, "data")

alpha = Path("configs/alpha_calibrated.json")
if not alpha.exists():
    print("regenerating the alpha table from the PCKPT P0 run", flush=True)
    subprocess.run([PY, "scripts/calibrate_alpha.py", "--checkpoint",
                    f"{DRIVE_RESULTS}/L4d256_1M_fp_backprop_K0_s0/best.pt",
                    "--batches", "8"], check=True)

# ---- 1. S probes per precision (probe 3 first on P3 — the H2 gate) ----
S_PCKPTS = [
    "L4d256_1M_w1p58a4_backprop_K0_s0",   # P3 primary: STE-bias gate first
    "L4d256_1M_fp_backprop_K0_s0",       # P0 reference
    "L4d256_1M_w1p58a8_backprop_K0_s0",  # P2
    "L4d256_1M_w8a8_backprop_K0_s0",     # P1
]
for run_id in S_PCKPTS:
    which = "3,2,5" if "w1p58a4" in run_id else "3,2"
    print(f"\n=== S probes {run_id} (probes {which}) ===", flush=True)
    subprocess.run([PY, "-m", "dustq.probes", "--config", f"configs/{run_id}.yaml",
                    "--probes", which], check=True)

# ---- 2. directional descent on S at P3 (cheap, method-neutral arbiter) ----
print("\n=== S directional descent (P3) ===", flush=True)
subprocess.run([PY, "-m", "dustq.probes", "--config",
                "configs/L4d256_1M_w1p58a4_backprop_K0_s0.yaml", "--probes", "4"],
               check=True)

# ---- 3. M spot-check at P3 (ladders only, unless the budget is comfortable) ----
M_P3 = "L8d512_1M_w1p58a4_dust_K256_s0"   # reuse a trained M checkpoint for the layers
if os.environ.get("M_SPOT", "1") == "1":
    print("\n=== M spot-check (ladders + sweep) ===", flush=True)
    subprocess.run([PY, "-m", "dustq.probes", "--config", f"configs/{M_P3}.yaml",
                    "--probes", "2,5"], check=True)

print(f"\nprobe session complete: {(time.monotonic() - T0) / 3600:.2f} h "
      "— copy results/probes/*.json back")
