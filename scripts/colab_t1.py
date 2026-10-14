"""Dust-Q T1 on Colab L4 — paste this whole file into ONE Colab cell and run it.

Before running:
  1. Runtime -> Change runtime type -> L4 GPU.
  2. Make a zip of the repo (excluding .git, .venv, results, data) and upload it via the
     Files panel on the left (or set REPO_ZIP to a Drive path / REPO_URL to a git remote).
  3. Optionally set CU_START from the Resources popup so we can estimate the burn rate.

It runs: benchmark (s/step + peak memory at real scale), the resume-exactness test on GPU,
and prints everything you need to paste back. Total ~20-40 L4 minutes.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ZIP = "/content/dustq.zip"          # uploaded via the Files panel
REPO_URL = ""                            # or a git remote, if you pushed the repo
CU_START = None                          # e.g. float("123.4") from the Colab Resources tab

# ---- 0. environment -------------------------------------------------------
import torch
T_CELL0 = time.monotonic()  # cell wall clock (M4: time.monotonic() alone is machine uptime)
print("torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "Restart with an L4 GPU runtime"
print("gpu:", torch.cuda.get_device_name(0), "| vram:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")

# ---- 1. get the code -------------------------------------------------------
if REPO_URL:
    subprocess.run(["git", "clone", REPO_URL, "dustq-repo"], check=True)
    os.chdir("dustq-repo")
else:
    assert Path(REPO_ZIP).exists(), f"Upload {REPO_ZIP} via the Files panel (or set REPO_URL)"
    import zipfile
    with zipfile.ZipFile(REPO_ZIP) as z:
        z.extractall("/content/dustq-repo")
    root = next(p for p in Path("/content/dustq-repo").iterdir() if (p / "dustq").is_dir())
    os.chdir(root)
print("repo at:", Path.cwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True)

# ---- 2. sanity: three parts, each with a distinct job ----------------------
# (i) code integrity (runs on CPU even on Colab): Phase 0 suite incl. the bitwise contract
subprocess.run([sys.executable, "-m", "pytest", "tests/test_phase0.py", "-q"], check=True)
# (ii) the CUDA path through Runner (the in-repo path that would have caught B4): resume subset
subprocess.run([sys.executable, "-m", "pytest", "tests/test_t1.py", "-k", "resume", "-q"], check=True)
# (iii) per-precision CUDA steps: one tiny DUST + one BP step on the GPU (the B4-catcher)
subprocess.run([sys.executable, "-m", "pytest", "tests/test_cuda.py", "-q"], check=True)
print("sanity ok (integrity + CUDA path + per-precision GPU steps)")

# ---- 3. benchmark at real scale -------------------------------------------
t_bench = time.monotonic()
subprocess.run([sys.executable, "scripts/benchmark.py", "--device", "cuda", "--scale", "real"], check=True)
print(f"\nbenchmark wall: {(time.monotonic() - t_bench)/60:.1f} min")

# ---- 4. resume-exactness test on GPU (fast subset) --------------------------
t_res = time.monotonic()
subprocess.run([sys.executable, "-m", "pytest", "tests/test_t1.py", "-k", "resume", "-q"], check=True)
print(f"resume test wall: {(time.monotonic() - t_res)/60:.1f} min")

# ---- 5. report -------------------------------------------------------------
print("\n" + Path("results/benchmark_cuda.md").read_text())
print("\nPaste the file contents above back to the agent, together with:")
print("  - CU/h from the Resources tab (and CU_START if you set it)",
      f"cell wall: {(time.monotonic() - T_CELL0) / 60:.0f} min")
if CU_START is not None:
    print("  (remember to read CU again now for the burn rate)")
print("\nOptional next: zip the repo and upload data files to Drive per docs/reports/T1.md.")
