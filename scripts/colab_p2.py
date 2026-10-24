"""Dust-Q Phase 2 on Colab L4 — paste this whole file into ONE cell and run.

Runs, in order (each step prints; failures stop the session with a clear message):
  0. GPU check + repo unzip + Drive data check + CUDA sanity tests
  1. Post-T5 benchmark (fp + w1.58a4 cells — measures the code-caching speedup; these
     numbers replace the stale pre-T5 quantized rates in the budget)
  2. S/1M backprop fp checkpoint (the PCKPT P0 config, ~5 min; doubles as the alpha_b
     calibration source)
  3. alpha_b calibration on that checkpoint -> writes configs/alpha_calibrated.json in
     the runtime copy + prints the JSON + saves a copy to Drive (paste it back so the
     canonical repo gets it)
  4. One queue pass: G1 dtype pairs + the remaining PCKPT runs (the P0 checkpoint is
     skipped as complete; P2/P3/G1-P3 configs resolve the table written in step 3)
  5. G1 single-batch gradient cosine: fp32 vs bf16 Dust gradients per layer type

Before running: L4 runtime; fresh dustq.zip uploaded; data on Drive at MyDrive/dustq/data
(already there). Expected: ~4-5 h, ~7 CU.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ZIP = "/content/dustq.zip"
DRIVE_DATA = "/content/drive/MyDrive/dustq/data"
DRIVE_RESULTS = "/content/drive/MyDrive/dustq/results"
CU_BURN_RATE = "1.54"

def run_checked(cmd, **kwargs):
    """subprocess.run with captured output; on failure, print the tails before raising —
    never let an inner crash's message vanish like the Tier-R OOM notes did."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print("COMMAND FAILED:", " ".join(cmd), flush=True)
        print("--- stdout tail ---\n" + result.stdout[-2500:], flush=True)
        print("--- stderr tail ---\n" + result.stderr[-4000:], flush=True)
        raise SystemExit(f"exit {result.returncode}; see tails above")
    if result.stdout.strip():
        print(result.stdout[-1500:], flush=True)
    return result


# ---- 0. environment ----
import torch
T_CELL0 = time.monotonic()
assert torch.cuda.is_available(), "Restart with an L4 GPU runtime"
print("gpu:", torch.cuda.get_device_name(0))

import zipfile
with zipfile.ZipFile(REPO_ZIP) as z:
    z.extractall("/content/dustq-repo")
base = Path("/content/dustq-repo")
root = next(p for p in [base] + [q for q in base.iterdir() if q.is_dir()] if (p / "dustq").is_dir())
os.chdir(root)
print("repo at:", root)
run_checked([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])

from google.colab import drive
drive.mount("/content/drive")
for name in ("train_1M.pt", "train_10M.pt", "val.pt", "test.pt"):
    assert (Path(DRIVE_DATA) / name).is_file(), f"missing {name} on Drive"
if not Path("data").exists():
    os.symlink(DRIVE_DATA, "data")
Path(DRIVE_RESULTS).mkdir(parents=True, exist_ok=True)
run_checked([sys.executable, "-m", "pytest", "tests/test_cuda.py", "-q"])
print("sanity ok")

# ---- 1. post-T5 benchmark (validates the code-caching speedup) ----
run_checked([sys.executable, "scripts/benchmark.py", "--device", "cuda", "--scale", "real",
             "--sizes", "S,M", "--populations", "256,1024", "--precisions", "fp,w1.58a4",
             "--steps", "2", "--skip-concurrent"])
print("PASTE the benchmark table back — it re-prices the CUT_CELLS restores.")

# ---- 2. S/1M backprop fp checkpoint (PCKPT P0) ----
sys.path.insert(0, str(root))
from dustq.config import load_config
from dustq.train import Runner
p0_cfg = load_config("configs/L4d256_1M_fp_backprop_K0_s0.yaml")
p0_cfg.output_dir = DRIVE_RESULTS
if not (Path(DRIVE_RESULTS) / p0_cfg.run_id / "result.json").exists():
    print("training", p0_cfg.run_id, flush=True)
    Runner(p0_cfg).train()
else:
    print("P0 checkpoint already complete")

# ---- 3. alpha_b calibration (Plan 4.2; frozen table; never learned) ----
alpha_path = Path("configs/alpha_calibrated.json")
if not alpha_path.exists():
    run_checked([sys.executable, "scripts/calibrate_alpha.py",
                 "--checkpoint", str(Path(DRIVE_RESULTS) / p0_cfg.run_id / "best.pt"),
                 "--batches", "8"])
print("PASTE this JSON back (canonical repo needs it):")
print(alpha_path.read_text())
Path("/content/drive/MyDrive/dustq/alpha_calibrated.json").write_text(alpha_path.read_text())

# ---- 4. queue: G1 dtype pairs + remaining PCKPTs (table now present in the runtime copy) ----
cmd = [sys.executable, "scripts/run_queue.py", "--configs-dir", "configs",
       "--results-dir", DRIVE_RESULTS, "--cu-burn-rate", CU_BURN_RATE, "--retry-failed"]
print(" ".join(cmd), flush=True)
run_checked(cmd)

# ---- 5. G1 single-batch gradient cosine: fp32 vs bf16 (Plan T4/G1) ----
from dustq import qmodel
from dustq.dust_ext import DUSTQ
from dustq.recipes import recipe
from dustq.runner import build_model, tiny_batch
from dust import load_sequences, make_optimizer

rows = load_sequences("data/train_1M.pt", 8)
x = rows[:, :-1].contiguous().cuda()
y = rows[:, 1:].contiguous().cuda()
cosines = {}
for dtype_name, dtype in (("bf16", None), ("fp32", torch.float32)):
    model = build_model(qmodel.SIZES["L8d512"], "fp", "cuda", seed=0).requires_grad_(False)
    settings = recipe("1M", 256, model.config.n_layer)
    est = DUSTQ(model, settings, seed=0)
    if dtype is not None:
        model.to(dtype=torch.float32)
        est.dtype = torch.float32
    est.step(x, y, make_optimizer(model, settings))
    for name, param in model.named_parameters():
        if param.grad is not None:
            cosines.setdefault(name, {})[dtype_name] = param.grad.float().flatten()
    est.close()
print("\nG1 gradient cosine, fp32 vs bf16 (Dust, M/1M, K=256, one batch):")
tot = {}
for name, grads in cosines.items():
    a, b = grads["fp32"], grads["bf16"]
    c = torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-12)
    tot.setdefault(name.split(".")[-2] if "." in name else name, []).append(c)
site_means = {k: sum(v) / len(v) for k, v in tot.items()}
for k, v in sorted(site_means.items()):
    print(f"  {k:16s} {v:+.4f}")
overall = sum(v for v in site_means.values()) / len(site_means)
print(f"  {'mean':16s} {overall:+.4f}  <- G1 P0 cosine (within seed noise = pass)")
print("\nSession complete. Paste back: benchmark table, alpha JSON, runs.csv rows, cosines.")
print(f"cell wall: {(time.monotonic() - T_CELL0) / 60:.0f} min")
