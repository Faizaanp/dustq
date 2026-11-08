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

# the checkpoints the probe session reads (must exist on Drive; preflight asserts it)
S_PCKPTS = [
    "L4d256_1M_w1p58a4_backprop_K0_s0",   # P3 primary
    "L4d256_1M_fp_backprop_K0_s0",       # P0 reference
    "L4d256_1M_w1p58a8_backprop_K0_s0",  # P2
    "L4d256_1M_w8a8_backprop_K0_s0",     # P1
]
M_P3 = "L8d512_1M_w1p58a4_dust_K256_s0"   # reuse a trained M checkpoint for the layers

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
# results/ is gitignored, so the fresh zip has none: without this link ProbeContext opens
# results/<run_id>/best.pt and every probe run dies with FileNotFoundError before its first
# gradient (this killed the 2026-11-02 session; the error was swallowed by subprocess.run).
if not Path("results").exists():
    os.symlink(DRIVE_RESULTS, "results")

def run_checked(cmd, **kwargs):
    """Capture output; on failure print the tails BEFORE exiting — never let an inner
    crash's message vanish (the Tier-R truncated-note lesson, applied here too late)."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print("COMMAND FAILED:", " ".join(cmd), flush=True)
        print("--- stdout tail ---\n" + result.stdout[-2500:], flush=True)
        print("--- stderr tail ---\n" + result.stderr[-4000:], flush=True)
        raise SystemExit(f"exit {result.returncode}; see tails above")
    if result.stdout.strip():
        print(result.stdout[-1500:], flush=True)
    return result


alpha = Path("configs/alpha_calibrated.json")
if not alpha.exists():
    print("regenerating the alpha table from the PCKPT P0 run", flush=True)
    subprocess.run([PY, "scripts/calibrate_alpha.py", "--checkpoint",
                    f"{DRIVE_RESULTS}/L4d256_1M_fp_backprop_K0_s0/best.pt",
                    "--batches", "8"], check=True)

# ---- 0a. fail fast on anything that would abort the session at its first probe ----
print("\n=== 0a. preflight (data, checkpoints, alpha table) ===", flush=True)
for need, label in ((Path("data/train_1M.pt"), "data/train_1M.pt"),
                    (Path("data/val.pt"), "data/val.pt"),
                    (Path("configs/alpha_calibrated.json"), "alpha_calibrated.json")):
    if not need.is_file():
        print(f"MISSING {label} — the probe session cannot start", flush=True)
        raise SystemExit(f"missing {label}")
for run_id in S_PCKPTS + [M_P3]:
    ckpt = Path("results") / run_id / "best.pt"
    print(f"{'ok ' if ckpt.is_file() else 'MISSING'} {ckpt}", flush=True)
missing_p = [r for r in S_PCKPTS + [M_P3] if not (Path("results") / r / "best.pt").is_file()]
if missing_p:
    raise SystemExit(f"checkpoints missing on Drive: {missing_p}")
print("preflight passed", flush=True)

# ---- 0b. G1 loss confirmation (the open half of gate G1) ----
print("\n=== G1 loss rows (paste these back) ===", flush=True)
import csv

runs_csv = Path(DRIVE_RESULTS) / "runs.csv"
if runs_csv.is_file():
    ids = [r for r in csv.DictReader(runs_csv.open())]
    wanted = ("L8d512_1M_fp_dust_K256_s0_fp32", "L8d512_1M_w1p58a4_dust_K256_s0",
              "L8d512_1M_w1p58a4_dust_K256_s0_fp32")
    hits = [r for r in ids if r["run_id"] in wanted]
    for r in hits:
        print(f"{r['run_id']}: val_best={r['val_loss_best']} test={r['test_loss_at_best_val']} "
              f"steps={r['steps']} status={r['status']}")
    if len(hits) < len(wanted):
        print(f"(only {len(hits)} of {len(set(wanted))} G1 rows found in runs.csv)")
    base = [r for r in ids if r["run_id"] in
            ("L8d512_1M_fp_dust_K256_s0", "L8d512_1M_fp_dust_K256_s1",
             "L8d512_1M_fp_dust_K256_s2") and r["status"] == "completed"]
    if base:
        print("bf16 K256 base (Tier R, for comparison):",
              [r["test_loss_at_best_val"] for r in base])
else:
    print("runs.csv not found on Drive")

# OQ-40 decision: the Dust probe gradients share the backprop probes' forward (Plan v3.1
# 4.2). "fake" is the plan-consistent default and ~1.4x faster for the estimator; "tf32"
# keeps the verified exact path for the ablation.
DUST_FORWARD = "fake"

# ---- 1. P3 FIRST, and the G2a arms before the expensive sweep ----
# G2a's criterion (Plan v3.1 section 5) is at P3: cos(Dust-1k, ref) RISES with K on MLP
# and writer layers, plus a negative directional descent along Dust-1k. So the ladder
# (probe 2) and the descent (probe 4) are the gate; the sigma-Delta sweep (probe 5) is
# the mechanism analysis and the most expensive step. Running the gate arms first - each
# in its own subprocess, each saving its own JSON - means a late failure costs only the
# mechanism data, not the gate reading. That is what cost half of G2a on 2026-11-02.
P3 = "L4d256_1M_w1p58a4_backprop_K0_s0"
# All five checkpoints, every probe. Deliberately NOT the "re-probe matrix": the five bugs
# fixed in R1-R4 all change the BASIS of a measurement (reference convention, group set,
# flip-rate definition, descent normalisation), so a matrix would mix pre- and post-fix
# numbers in the same comparison. One uniform session gives one internally consistent set.
# The pre-fix JSONs stay on Drive as the audit-trail cohort and nothing new is compared
# to them. P0's probe 3 is the one redundancy (at fp there is no activation grid, so SSTE
# is bit-identical to STE) and costs ~2 min, so it stays for uniformity.
STEPS = [(P3, "3,2,0"), (P3, "4"), (P3, "5")]
STEPS += [(r, "3,2") for r in S_PCKPTS if r != P3]      # P0, P2, P1
STEPS.append((M_P3, "2,5"))

PROBE_DIR = Path("results/probes")


def json_path(run_id: str, which: str) -> Path:
    """The filename dustq.probes writes for this subset (same tag logic as probes.main)."""
    tag = "-".join(str(p) for p in sorted(int(p) for p in which.split(",")))
    return PROBE_DIR / f"{run_id}_probes_{tag}.json"


print("\n=== resume plan (done steps are skipped) ===", flush=True)
todo = []
for run_id, which in STEPS:
    done = json_path(run_id, which).is_file()
    print(f"  {'DONE ' if done else 'TODO '} {run_id} probes {which}", flush=True)
    if not done:
        todo.append((run_id, which))

for run_id, which in todo:
    print(f"\n=== runs {run_id} (probes {which}, forward={DUST_FORWARD}) ===", flush=True)
    run_checked([PY, "-m", "dustq.probes", "--config", f"configs/{run_id}.yaml",
                 "--probes", which, "--dust-forward", DUST_FORWARD])
    written = json_path(run_id, which)
    print(f"  -> {written} ({written.stat().st_size if written.is_file() else 'MISSING'} bytes)",
          flush=True)
    if not written.is_file():
        print("  !! step reported success but wrote no JSON - investigate before continuing",
              flush=True)

skipped = len(STEPS) - len(todo)
print(f"\nprobe session complete: {(time.monotonic() - T0) / 3600:.2f} h "
      f"({len(todo)} run, {skipped} skipped as already done)")

# ---- 3. M spot-check at P3 (ladders + sweep, if the budget is comfortable) ----


