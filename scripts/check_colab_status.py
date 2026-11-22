"""Dust-Q Colab Telemetry & Status Checker (Non-Blocking).

Queries the active Google Colab L4 assignment, refreshes local session proxy tokens,
and downloads the latest probe logs, tuning logs, and runs.csv from Drive via the
Colab runtime proxy HTTP contents API.

CRITICAL OPERATIONAL GUARANTEE:
This script NEVER calls `colab.exe exec` while long training or probing jobs are running,
avoiding the Jupyter kernel execution serialization deadlock.
"""

import os
import subprocess
import sys
from pathlib import Path

# Base directories
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from colab_cli.common import state
except ImportError:
    print("google-colab-cli not installed in python environment.", file=sys.stderr)
    sys.exit(1)


def refresh_token(session_name: str = "tune-session"):
    """Auto-refresh Colab proxy token in state.store if renewed by Google."""
    assignments = state.client.list_assignments()
    if not assignments:
        print("[status] No active Colab assignments found on account.")
        return None
    a = assignments[0]
    s = state.store.get(session_name)
    if s and a.runtime_proxy_info.token != s.token:
        s.token = a.runtime_proxy_info.token
        s.token_expires_at = a.runtime_proxy_info.expires_at()
        s.url = a.runtime_proxy_info.url
        s.endpoint = a.endpoint
        state.store.add(s)
        print(f"[status] Updated proxy token for {session_name} (endpoint: {a.endpoint}).")
    elif s:
        print(f"[status] Session {session_name} active on {a.endpoint} ({a.accelerator}).")
    return s


def check_telemetry(session_name: str = "tune-session"):
    colab_bin = REPO_ROOT / ".venv" / "Scripts" / "colab.exe"
    if not colab_bin.is_file():
        colab_bin = Path("colab")

    tmp_dir = RESULTS_DIR / "_telemetry_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and parse probe log
    probe_log = tmp_dir / "colab_probe.log"
    res_p = subprocess.run([str(colab_bin), "download", "-s", session_name,
                            "/content/drive/MyDrive/dustq/results/colab_probe.log", str(probe_log)],
                           capture_output=True, text=True)
    if probe_log.is_file():
        lines = probe_log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        print(f"\n=== Probe Log Tail ({len(lines)} total lines) ===")
        for l in lines[-20:]:
            print(" ", l)
        probe_log.unlink(missing_ok=True)
    else:
        print("\n[status] No active colab_probe.log found on Drive.")

    # 2. Download and parse tune log
    tune_log = tmp_dir / "colab_tune.log"
    res_t = subprocess.run([str(colab_bin), "download", "-s", session_name,
                            "/content/drive/MyDrive/dustq/results/colab_tune.log", str(tune_log)],
                           capture_output=True, text=True)
    if tune_log.is_file():
        lines = tune_log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        print(f"\n=== Tune Log Tail ({len(lines)} total lines) ===")
        for l in lines[-15:]:
            print(" ", l)
        tune_log.unlink(missing_ok=True)

    # 3. Download and parse runs.csv
    runs_csv = tmp_dir / "runs.csv"
    res_r = subprocess.run([str(colab_bin), "download", "-s", session_name,
                            "/content/drive/MyDrive/dustq/results/runs.csv", str(runs_csv)],
                           capture_output=True, text=True)
    if runs_csv.is_file():
        lines = runs_csv.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        tune_lines = [l for l in lines[1:] if "tune" in l]
        print(f"\n=== Runs CSV Summary: {len(tune_lines)} / 114 tuning runs completed ===")
        for l in lines[-5:]:
            p = l.split(",")
            val_loss = p[14] if len(p) > 14 else "?"
            train_loss = p[15] if len(p) > 15 else "?"
            print(f"  {p[0]} | val={val_loss} | train={train_loss}")
        runs_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    refresh_token("tune-session")
    check_telemetry("tune-session")
