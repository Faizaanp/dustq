"""Dust-Q T6 probe session on Colab L4 — paste into ONE cell and run.

Requires: a FRESH dustq.zip (rebuild it after every commit: `git archive -o dustq.zip HEAD`),
an L4 runtime, and Drive data at MyDrive/dustq/data.

Runs, in order:
  P3 (w1p58a4, the gate precision): probes 3,2,0 then 4 then 5 — each its own subprocess so
      a late failure costs only the mechanism data, not the gate reading.
  P0 (fp), P2 (w1.58a8), P1 (w8a8): probes 3,2 — the cross-precision comparison for the paper.
  M (L8d512, w1.58a4): probes 2,5 — the layers spot-check.

Outputs: results/probes/<run_id>_probes_<tag>.json, one per (checkpoint, probe-set), also
written to Drive. Completed steps are SKIPPED on re-run, so a disconnected session resumes
rather than restarting. Expected ~3 h, ~5-6 CU at the owner-observed 1.94 CU/h.

After the 2026-11-08 audit fixes (R1-R4), nothing from an earlier JSON is comparable: the
reference convention, the group set, the flip-rate basis and the descent normalisation all
changed. This session produces the first internally consistent set.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ZIP = "/content/dustq.zip"
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
print("repo at:", root)

if not (Path("/content/drive/MyDrive").is_dir() or Path("/content/drive/My Drive").is_dir()):
    from google.colab import drive
    drive.mount("/content/drive")

# Support both /content/drive/MyDrive and /content/drive/My Drive
drive_candidates = [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]
drive_root = next((p for p in drive_candidates if p.is_dir()), None)
if drive_root is None:
    raise SystemExit("Neither /content/drive/MyDrive nor /content/drive/My Drive found!")

DRIVE_DATA = drive_root / "dustq" / "data"
DRIVE_RESULTS = drive_root / "dustq" / "results"
DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)
(DRIVE_RESULTS / "probes").mkdir(parents=True, exist_ok=True)

# Robust symlinking: os.path.lexists / unlink prevents FileExistsError on broken symlinks
for link_name, target in [("data", DRIVE_DATA), ("results", DRIVE_RESULTS)]:
    p = Path(link_name)
    if os.path.lexists(link_name):
        if p.is_symlink() and p.resolve() == target.resolve() and p.exists():
            continue
        p.unlink(missing_ok=True)
    if not p.exists():
        os.symlink(target, link_name)

Path("results/probes").mkdir(parents=True, exist_ok=True)

def run_checked(cmd, **kwargs):
    """Stream stdout/stderr line-by-line so Colab never appears frozen or stalls
    silently for hours, while retaining full tails for actionable crash diagnostics."""
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


alpha = Path("configs/alpha_calibrated.json")
if not alpha.exists():
    print("regenerating the alpha table from the PCKPT P0 run", flush=True)
    run_checked([PY, "scripts/calibrate_alpha.py", "--checkpoint",
                 f"{DRIVE_RESULTS}/L4d256_1M_fp_backprop_K0_s0/best.pt",
                 "--batches", "8"])

# ---- 0a. fail fast on anything that would abort the session at its first probe ----
print("\n=== 0a. preflight (data, checkpoints, alpha table) ===", flush=True)
for need, label in ((Path("data/train_1M.pt"), "data/train_1M.pt"),
                    (Path("data/val.pt"), "data/val.pt"),
                    (Path("configs/alpha_calibrated.json"), "alpha_calibrated.json")):
    if not need.is_file():
        print(f"MISSING {label} — the probe session cannot start", flush=True)
        raise SystemExit(f"missing {label}")

import json as _json
_alpha_data = _json.loads(Path("configs/alpha_calibrated.json").read_text())
_alphas = _alpha_data.get("alpha") or {}
_missing_alpha = [k for k in ("a4s", "a4u", "a2s", "a2u") if k not in _alphas]
if _missing_alpha:
    raise SystemExit(f"configs/alpha_calibrated.json missing required 'alpha' keys: {_missing_alpha}")

missing_p = []
for run_id in S_PCKPTS + [M_P3]:
    ckpt = Path("results") / run_id / "best.pt"
    exists = ckpt.is_file()
    status = "ok     " if exists else "MISSING"
    diag = ""
    if not exists:
        missing_p.append(run_id)
        run_dir = Path("results") / run_id
        if (run_dir / "ckpt.pt").is_file():
            diag = " (partial run: ckpt.pt exists, resume via scripts/run_queue.py)"
        elif (run_dir / "FAILED_ATTEMPTS").is_file():
            note = (run_dir / "FAILED_ATTEMPTS").read_text().splitlines()
            diag = f" (failed run: {note[-1] if note else 'unknown'})"
        elif not run_dir.exists():
            diag = " (directory does not exist on Drive)"
    print(f"{status} {ckpt}{diag}", flush=True)

if missing_p:
    print(f"\nActionable fix: run the queue to generate missing checkpoints on Colab L4:", flush=True)
    for r in missing_p:
        print(f"  {PY} scripts/run_queue.py --only {r}", flush=True)
    raise SystemExit(f"checkpoints missing on Drive ({DRIVE_RESULTS}): {missing_p}")
print("preflight passed", flush=True)

# ---- 0b. G1 loss confirmation (the open half of gate G1) ----
print("\n=== G1 loss rows (paste these back) ===", flush=True)
import csv

runs_csv = Path(DRIVE_RESULTS) / "runs.csv"
if runs_csv.is_file():
    ids = [r for r in csv.DictReader(runs_csv.open())]
    wanted = ("L8d512_1M_fp_dust_K256_s0_fp32", "L8d512_1M_w1p58a4_dust_K256_s0",
              "L8d512_1M_w1p58a4_dust_K256_s0_fp32")
    hits = [r for r in ids if r.get("run_id") in wanted]
    for r in hits:
        print(f"{r.get('run_id')}: val_best={r.get('val_loss_best')} test={r.get('test_loss_at_best_val')} "
              f"steps={r.get('steps')} status={r.get('status')}")
    if len(hits) < len(wanted):
        print(f"(only {len(hits)} of {len(set(wanted))} G1 rows found in runs.csv)")
    base = [r for r in ids if r.get("run_id") in
            ("L8d512_1M_fp_dust_K256_s0", "L8d512_1M_fp_dust_K256_s1",
             "L8d512_1M_fp_dust_K256_s2") and r.get("status") == "completed"]
    if base:
        print("bf16 K256 base (Tier R, for comparison):",
              [r.get("test_loss_at_best_val") for r in base])
else:
    print("runs.csv not found on Drive")

# OQ-40 decision: the Dust probe gradients share the backprop probes' forward (Plan v3.1
# 4.2). "fake" is the plan-consistent default and ~1.4x faster for the estimator; "tf32"
# keeps the verified exact path for the ablation.
DUST_FORWARD = os.environ.get("DUST_FORWARD", "fake")

# ---- the run plan ----
# P3 is the G2a gate precision, and its gate arms (probe 2's ladder, probe 4's descent) run
# before the expensive sigma-Delta sweep, each as its own subprocess saving its own JSON -
# so a late failure costs only the mechanism data, not the gate reading. That ordering is
# what cost half of G2a on 2026-11-02.
P3 = "L4d256_1M_w1p58a4_backprop_K0_s0"

# All five checkpoints, every probe. Deliberately NOT the "re-probe matrix": the R1-R4 fixes
# each change the BASIS of a measurement (reference convention, group set, flip-rate basis,
# descent normalisation), so a matrix would mix pre- and post-fix numbers in one comparison.
# One uniform session gives one internally consistent set; the pre-fix JSONs stay on Drive as
# the audit-trail cohort and nothing new is compared to them. P0's probe 3 is the one
# redundancy (at fp there is no activation grid, so SSTE is bit-identical to STE) and costs
# about two minutes, so it stays for uniformity.
STEPS = [(P3, "3,2,0"), (P3, "4"), (P3, "5")]
STEPS += [(r, "3,2") for r in S_PCKPTS if r != P3]      # P0, P2, P1
STEPS.append((M_P3, "2,5"))                            # M
STEPS.append((M_P3, "4"))                              # M Arm 2 directional descent

PROBE_DIR = Path("results/probes")
# Pass FORCE_PROBES=4 (or 2,4) in env only when intentionally refreshing already-completed probes.
FORCE_PROBES = {p.strip() for p in os.environ.get("FORCE_PROBES", "").split(",") if p.strip()}


def json_path(run_id: str, which: str) -> Path:
    """The post-override filename dustq.probes writes for this config and subset."""
    from dustq.config import load_config
    from dustq.probes import probe_report_path

    cfg = load_config(f"configs/{run_id}.yaml")
    if DUST_FORWARD:
        cfg.dust_forward = DUST_FORWARD
    return probe_report_path(cfg, which, PROBE_DIR)


SECTION_MAP = {
    0: "seed_noise",
    2: "cosine_ladders",
    3: "ste_bias",
    4: "directional_descent",
    5: "sigma_delta",
}


def is_step_done(run_id: str, which: str) -> bool:
    """R14 + F2/O2: Validate section presence AND schema_version >= required minimum before skipping."""
    from dustq.probes import is_probe_report_current

    step_probes = {p.strip() for p in which.split(",")}
    if any(p in FORCE_PROBES for p in step_probes):
        return False
    p = json_path(run_id, which)
    return is_probe_report_current(p, which)


print("\n=== resume plan (done steps are skipped; stale pre-v2 probe 2/4 JSONs re-run automatically) ===", flush=True)
todo = []
for run_id, which in STEPS:
    done = is_step_done(run_id, which)
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

# Optional Option-B chaining: set AUTO_RESUME_T7=1 to immediately resume the T7 tuning queue
# after the clean probe refresh completes.
if os.environ.get("AUTO_RESUME_T7", "0") == "1":
    print("\n=== AUTO_RESUME_T7=1: Chaining into Task T7 Tuning Queue ===", flush=True)
    run_checked([PY, "scripts/make_configs.py", "--tier", "tune", "--out", "configs"])
    filter_substr = os.environ.get("TUNE_ONLY", "_tune")
    run_checked([PY, "scripts/run_queue.py", "--only", filter_substr, "--cu-burn-rate", "1.94"])
    run_checked([PY, "scripts/analyze_tuning.py", "--results-dir", "results", "--out", "results/tune_best_hparams.json"])

# ---- what to read when it finishes ----
# G2a's criterion (Plan v3.1 section 5) is P3-internal and has two arms:
#   arm 1: cos(Dust-K, ref) RISES with K on 'MLP in', 'MLP out' and 'Attn out'
#          (the writer/hub groups), read from the 'cosine_ladders' section
#   arm 2: directional descent along Dust-1k is NEGATIVE on both train and val,
#          read from the 'directional_descent' section ('Dust1k' entry)
# The disattenuated column divides by sqrt(spearman_brown(r_g)) PER GROUP and is None
# wherever r_g <= 0 (undefined, not a number); the JSON's 'disattenuation' records both.
# NOTHING from a pre-2026-11-08 JSON is comparable to this run: the reference convention,
# the group set, the flip-rate basis and the descent normalisation all changed.
print("""
=== reading the results ===
  arm 1  ladder must RISE with K on 'MLP in'/'MLP out'/'Attn out'   (cosine_ladders)
  arm 2  'Dust1k' train+val descent must be NEGATIVE                (directional_descent)
  also   seed-noise floor per group (probe 0) - the variance baseline the attention-
         internal cosines must be judged against
         sigma-Delta per-group cosines (probe 5) - already show q/k/v are sigma-INSENSITIVE
         while the writer groups peak at ~1x Delta
""", flush=True)
print("copy results/probes/*_probes_*.json back - they are also saved on Drive", flush=True)
