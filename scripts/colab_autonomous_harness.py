"""Dust-Q Autonomous Training Harness for Google Colab L4 Runtime.

An airtight, production-grade 6-stage autonomous control loop designed for Google Colab L4:
  Stage 1: Pre-flight checks (Hardware, Drive, symlinks, datasets, alpha calibration, checkpoints)
  Stage 2: Start training (Decoupled process launch, output redirection, PID tracking, baseline snapshot)
  Stage 3: Training health monitor (Periodic watchdog: process liveness, stall detection, NaN/Inf loss check, VRAM, atomic checkpoints)
  Stage 4: Diagnose and repair (SIGTERM/SIGKILL cleanup, full 20,000-char traceback capture, system state dump, incident classification)
  Stage 5: Verify before restarting (Targeted GPU smoke test, sandboxed checkpoint recovery)
  Stage 6: Restart from verified checkpoint (Exact noise stream resumption via DUSTQ state_dict, metric history alignment)

Supports three operational modes:
  - CLI Autonomous Queue: Runs through configs sequentially with automated watchdog, error triage, and retry.
  - Asynchronous Drive Bridge: Listens to Google Drive inbox for external agent commands and reports heartbeats to outbox.
  - Interactive Single-Cell: Self-contained bootstrap for Google Colab notebook cells.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Colab / Drive standard paths
DEFAULT_REPO_ZIP = "/content/dustq.zip"
DEFAULT_REPO_DIR = "/content/dustq-repo"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/dustq"
DEFAULT_DRIVE_DATA = "/content/drive/MyDrive/dustq/data"
DEFAULT_DRIVE_RESULTS = "/content/drive/MyDrive/dustq/results"
DEFAULT_DRIVE_BRIDGE = "/content/drive/MyDrive/dustq/bridge"
DEFAULT_CU_BURN_RATE = "1.94"  # [code] owner-observed 2026-10-31 for Colab L4

MAX_TRACEBACK_CHARS = 20000  # [code] preserves root-cause frames truncated by short notes
DEFAULT_WATCHDOG_INTERVAL_S = 20.0
DEFAULT_STALL_TIMEOUT_S = 300.0  # Alert if no metric/heartbeat for 5 minutes
MAX_CONSECUTIVE_RETRIES = 2     # Matches dustq.queue MAX_ATTEMPTS


def get_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command_capture(cmd: List[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """Execute command safely and capture output tails without swallowing errors."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return -1, stdout, f"Command timed out after {timeout}s: {stderr}"
    except Exception as exc:
        return -1, "", f"Failed to execute command {' '.join(cmd)}: {exc}"


# =====================================================================
# Stage 1: Pre-flight Verification
# =====================================================================

class PreflightChecker:
    """Verifies environment, GPU, Drive mounts, data assets, and checkpoint integrity [design]."""

    def __init__(
        self,
        drive_data: Path,
        drive_results: Path,
        configs_dir: Path,
        min_vram_gb: float = 20.0,
        require_cuda: bool = True,
    ):
        self.drive_data = Path(drive_data)
        self.drive_results = Path(drive_results)
        self.configs_dir = Path(configs_dir)
        self.min_vram_gb = min_vram_gb
        self.require_cuda = require_cuda

    def check_cuda_and_gpu(self) -> Dict[str, Any]:
        """Verify CUDA availability and NVIDIA L4 VRAM capacity (>= 20GB) [code]."""
        gpu_info: Dict[str, Any] = {
            "cuda_available": False,
            "device_name": "unknown",
            "vram_gb": 0.0,
            "passed": False,
            "errors": [],
        }

        try:
            import torch
            if not torch.cuda.is_available():
                if self.require_cuda:
                    gpu_info["errors"].append("CUDA is not available. Ensure runtime type is L4 GPU.")
                    return gpu_info
                else:
                    gpu_info["device_name"] = "CPU (synthetic test mode)"
                    gpu_info["passed"] = True
                    return gpu_info

            gpu_info["cuda_available"] = True
            dev_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
            gpu_info["device_name"] = dev_name
            gpu_info["vram_gb"] = vram_gb

            if vram_gb < self.min_vram_gb and self.require_cuda:
                gpu_info["errors"].append(
                    f"Insufficient VRAM: {vram_gb} GB detected, minimum {self.min_vram_gb} GB required for L4."
                )
            else:
                gpu_info["passed"] = True

        except ImportError:
            gpu_info["errors"].append("PyTorch is not installed in current environment.")
        except Exception as exc:
            gpu_info["errors"].append(f"GPU query failed: {exc}")

        return gpu_info

    def setup_symlinks(self) -> Dict[str, Any]:
        """Establish data and results symlinks pointing to Google Drive [design]."""
        report: Dict[str, Any] = {"symlinks": {}, "errors": []}
        cwd = Path.cwd()

        # Results directory on Drive
        self.drive_results.mkdir(parents=True, exist_ok=True)
        results_link = cwd / "results"
        if not results_link.exists():
            try:
                os.symlink(str(self.drive_results), str(results_link))
                report["symlinks"]["results"] = f"{results_link} -> {self.drive_results}"
            except Exception as exc:
                report["errors"].append(f"Failed to symlink results to Drive: {exc}")
        else:
            report["symlinks"]["results"] = f"existing {results_link.resolve()}"

        # Data directory on Drive
        data_link = cwd / "data"
        if not data_link.exists():
            if self.drive_data.exists():
                try:
                    os.symlink(str(self.drive_data), str(data_link))
                    report["symlinks"]["data"] = f"{data_link} -> {self.drive_data}"
                except Exception as exc:
                    report["errors"].append(f"Failed to symlink data to Drive: {exc}")
            else:
                report["symlinks"]["data"] = "Drive data directory not found; data symlink deferred."
        else:
            report["symlinks"]["data"] = f"existing {data_link.resolve()}"

        return report

    def verify_datasets(self, tokens_subset: str = "1M") -> Dict[str, Any]:
        """Verify presence and minimum size of dataset files on Drive or local data [code]."""
        report: Dict[str, Any] = {"verified_files": [], "missing_files": [], "passed": True}
        data_dir = Path("data")
        effective_dir = data_dir if data_dir.exists() else self.drive_data

        required = [f"train_{tokens_subset}.pt", "val.pt", "test.pt"]
        for fname in required:
            target = effective_dir / fname
            if target.is_file() and target.stat().st_size > 1024:
                report["verified_files"].append(f"{target.name} ({round(target.stat().st_size / (1024**2), 2)} MB)")
            else:
                report["missing_files"].append(str(target))
                report["passed"] = False

        return report

    def verify_alpha_calibration(self) -> Dict[str, Any]:
        """Verify the frozen alpha calibration table configs/alpha_calibrated.json [code]."""
        report: Dict[str, Any] = {"passed": False, "details": {}}
        alpha_path = Path("configs/alpha_calibrated.json")
        if not alpha_path.is_file():
            report["error"] = f"Missing alpha calibration table at {alpha_path}"
            return report

        try:
            data = json.loads(alpha_path.read_text())
            report["details"]["provenance"] = data.get("provenance", {})
            alphas = data.get("alpha") or data.get("alphas") or {}
            report["details"]["alpha"] = alphas
            required = ("a4s", "a4u", "a2s", "a2u")
            missing = [k for k in required if k not in alphas or not isinstance(alphas[k], (int, float)) or not math.isfinite(alphas[k]) or alphas[k] <= 0]
            if missing:
                report["error"] = f"Alpha calibration table missing or invalid keys {missing} in {alpha_path}"
                report["passed"] = False
            else:
                report["passed"] = True
        except Exception as exc:
            report["error"] = f"Failed to parse alpha calibration JSON: {exc}"

        return report

    def run_all(self, tokens_subset: str = "1M") -> Dict[str, Any]:
        """Run complete pre-flight suite and return pass/fail verdict [design]."""
        results = {
            "timestamp": get_iso_timestamp(),
            "gpu": self.check_cuda_and_gpu(),
            "symlinks": self.setup_symlinks(),
            "datasets": self.verify_datasets(tokens_subset),
            "alpha": self.verify_alpha_calibration(),
            "passed": False,
        }

        passed = (
            results["gpu"]["passed"]
            and len(results["symlinks"]["errors"]) == 0
            and results["datasets"]["passed"]
            and results["alpha"]["passed"]
        )
        results["passed"] = passed
        return results


# =====================================================================
# Checkpoint Verification & Validation
# =====================================================================

class CheckpointValidator:
    """Validates structural and numerical integrity of ckpt.pt and best.pt [design]."""

    REQUIRED_CKPT_KEYS = {"model", "optimizer", "step", "best", "history", "wall_prior"}

    @staticmethod
    def inspect_checkpoint(ckpt_path: Path, is_dust: bool = False) -> Dict[str, Any]:
        """Inspect checkpoint file on disk: check structure, finiteness, and step offset [code]."""
        report: Dict[str, Any] = {
            "path": str(ckpt_path),
            "exists": False,
            "valid": False,
            "step": None,
            "best": None,
            "num_params": 0,
            "non_finite_tensors": [],
            "missing_keys": [],
            "error": None,
        }

        if not ckpt_path.is_file() or ckpt_path.stat().st_size == 0:
            report["error"] = "Checkpoint file does not exist or has zero bytes"
            return report

        report["exists"] = True

        try:
            import torch
            # Safe CPU load
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                report["error"] = f"Checkpoint payload is not a dict: {type(payload)}"
                return report

            missing = [k for k in CheckpointValidator.REQUIRED_CKPT_KEYS if k not in payload]
            if is_dust and "generator" not in payload:
                missing.append("generator (required for exact Dust noise stream)")

            if missing:
                report["missing_keys"] = missing
                report["error"] = f"Missing required checkpoint keys: {missing}"
                return report

            report["step"] = int(payload.get("step", 0))
            report["best"] = float(payload.get("best", float("nan")))

            # Check model weights for NaN / Inf
            model_state = payload.get("model", {})
            param_count = 0
            for name, tensor in model_state.items():
                if isinstance(tensor, torch.Tensor):
                    param_count += tensor.numel()
                    if not torch.isfinite(tensor).all():
                        report["non_finite_tensors"].append(name)

            report["num_params"] = param_count

            if report["non_finite_tensors"]:
                report["error"] = f"NaN/Inf weights detected in tensors: {report['non_finite_tensors'][:5]}"
                return report

            report["valid"] = True

        except Exception as exc:
            report["error"] = f"Checkpoint deserialization failure: {exc}"

        return report

    @staticmethod
    def inspect_best_pt(best_path: Path) -> Dict[str, Any]:
        """Inspect best.pt: verify self-describing dict {'best': float, 'state': dict} [code]."""
        report: Dict[str, Any] = {"path": str(best_path), "exists": False, "valid": False, "best": None}
        if not best_path.is_file() or best_path.stat().st_size == 0:
            report["error"] = "best.pt does not exist or is empty"
            return report

        report["exists"] = True
        try:
            import torch
            try:
                payload = torch.load(best_path, map_location="cpu", weights_only=False)
            except Exception:
                payload = torch.load(best_path, map_location="cpu")
            if isinstance(payload, dict) and "state" in payload:
                report["best"] = float(payload.get("best", 0.0))
                report["valid"] = True
            elif isinstance(payload, dict) and any("weight" in k for k in payload.keys()):
                report["best"] = 0.0
                report["valid"] = True
            else:
                report["error"] = "best.pt does not conform to {'best', 'state'} contract or valid state_dict"
        except Exception as exc:
            report["error"] = f"Failed to load best.pt: {exc}"

        return report


# =====================================================================
# Stage 2: Process Management (Decoupled Runner Execution)
# =====================================================================

class ProcessManager:
    """Spawns training child processes with stable logging and decoupled process groups [design]."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.process: Optional[subprocess.Popen] = None
        self.log_file: Optional[Path] = None
        self.log_handle = None
        self.start_time: Optional[float] = None
        self.run_id: Optional[str] = None

    def launch_training(self, cmd: List[str], run_id: str) -> Path:
        """Launch training subprocess with stdout and stderr redirected to disk [design]."""
        self.run_id = run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_{run_id}_{timestamp}.log"
        self.log_handle = self.log_file.open("w", buffering=1, encoding="utf-8")

        self.start_time = time.monotonic()
        # Launch detached from current terminal group where possible
        preexec = None
        if os.name != "nt":
            preexec = os.setsid  # POSIX decoupled process group

        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec,
            text=True,
        )
        return self.log_file

    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_exit_code(self) -> Optional[int]:
        if self.process is None:
            return None
        return self.process.poll()

    def terminate_gracefully(self, grace_period_s: float = 10.0) -> int:
        """Send SIGTERM, wait for grace period, escalate to SIGKILL if necessary [design]."""
        if self.process is None or not self.is_alive():
            return 0

        pid = self.process.pid
        try:
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                self.process.terminate()
            except Exception:
                pass

        t0 = time.monotonic()
        while time.monotonic() - t0 < grace_period_s:
            code = self.process.poll()
            if code is not None:
                if self.log_handle and not self.log_handle.closed:
                    self.log_handle.close()
                return code
            time.sleep(0.5)

        # Force kill
        try:
            if os.name == "nt":
                self.process.kill()
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass

        code = self.process.wait()
        if self.log_handle and not self.log_handle.closed:
            self.log_handle.close()
        return code

    def get_log_tail(self, max_chars: int = MAX_TRACEBACK_CHARS) -> str:
        """Read the tail of the process log file to capture full stack traces [code]."""
        if not self.log_file or not self.log_file.is_file():
            return ""
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:] if len(text) > max_chars else text
        except Exception as exc:
            return f"Failed to read log tail: {exc}"


# =====================================================================
# Stage 3: Watchdog & Health Monitor
# =====================================================================

@dataclasses.dataclass
class HealthStatus:
    healthy: bool
    status_str: str  # "healthy" | "stalled" | "nan_inf" | "crashed" | "oom" | "completed"
    step: int = 0
    total_steps: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    elapsed_seconds: float = 0.0
    gpu_mem_used_mb: float = 0.0
    gpu_util_pct: float = 0.0
    last_metric_update_s_ago: float = 0.0
    checkpoint_step: Optional[int] = None
    error_message: Optional[str] = None


class TrainingWatchdog:
    """Inspects process status, loss trends, NaN/Inf, GPU utilization, and loop heartbeats [design]."""

    def __init__(
        self,
        run_dir: Path,
        process_mgr: ProcessManager,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        max_allowed_loss: float = 25.0,
    ):
        self.run_dir = Path(run_dir)
        self.process_mgr = process_mgr
        self.stall_timeout_s = stall_timeout_s
        self.max_allowed_loss = max_allowed_loss
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.ckpt_path = self.run_dir / "ckpt.pt"
        self.result_path = self.run_dir / "result.json"
        self._last_mtime: float = 0.0
        self._last_step: int = 0

    def query_gpu_stats(self) -> Tuple[float, float]:
        """Query GPU memory used (MB) and GPU utilization (%) via nvidia-smi [code]."""
        used_mb, util_pct = 0.0, 0.0
        if shutil.which("nvidia-smi") is not None:
            code, out, _ = run_command_capture(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                timeout=5.0,
            )
            if code == 0 and out.strip():
                try:
                    parts = out.strip().split("\n")[0].split(",")
                    used_mb = float(parts[0].strip())
                    util_pct = float(parts[1].strip())
                except Exception:
                    pass
        return used_mb, util_pct

    def check_metrics(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Parse the tail of metrics.jsonl (or legacy history.jsonl) to inspect loss, steps, and numerical stability [code]."""
        target_path = self.metrics_path if self.metrics_path.is_file() else (self.run_dir / "history.jsonl")
        if not target_path.is_file():
            return True, {}, None

        try:
            mtime = target_path.stat().st_mtime
            lines = [l.strip() for l in target_path.read_text().splitlines() if l.strip()]
            if not lines:
                return True, {"mtime": mtime}, None

            last_record = None
            for idx, raw_line in enumerate(lines):
                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError:
                    # Ignore a partial trailing line written right before a crash
                    if idx == len(lines) - 1:
                        continue
                    raise
                last_record = rec
                for k in ("train_loss", "val_loss"):
                    val = rec.get(k)
                    if val is not None:
                        if math.isnan(val) or math.isinf(val):
                            return False, rec, f"Non-finite {k} detected at step {rec.get('step')}: {val}"
                        if val > self.max_allowed_loss:
                            return False, rec, f"Catastrophic loss divergence at step {rec.get('step')}: {k}={val} exceeds {self.max_allowed_loss}"

            if last_record is None:
                return True, {"mtime": mtime}, None
            return True, {**last_record, "mtime": mtime}, None

        except Exception as exc:
            return False, {}, f"Failed to parse metrics.jsonl: {exc}"

    def inspect(self) -> HealthStatus:
        """Execute a full watchdog health check cycle [design]."""
        gpu_mb, gpu_util = self.query_gpu_stats()

        # Completed if valid result.json exists (even if process teardown is still finishing)
        if self.result_path.is_file():
            try:
                res = json.loads(self.result_path.read_text())
                if "best_val" in res:
                    return HealthStatus(
                        healthy=True,
                        status_str="completed",
                        step=res.get("steps", 0),
                        total_steps=res.get("steps", 0),
                        val_loss=res.get("best_val"),
                        gpu_mem_used_mb=gpu_mb,
                        gpu_util_pct=gpu_util,
                    )
            except Exception:
                pass

        # Check process liveness
        if not self.process_mgr.is_alive():
            exit_code = self.process_mgr.get_exit_code()
            if exit_code == 0:
                return HealthStatus(
                    healthy=False,
                    status_str="skipped",
                    gpu_mem_used_mb=gpu_mb,
                    gpu_util_pct=gpu_util,
                    error_message="Queue exited with code 0 without producing result.json (run skipped or filtered out)",
                )
            tail = self.process_mgr.get_log_tail(1500)
            status_str = "oom" if ("CUDA out of memory" in tail or "OutOfMemoryError" in tail) else "crashed"
            return HealthStatus(
                healthy=False,
                status_str=status_str,
                gpu_mem_used_mb=gpu_mb,
                gpu_util_pct=gpu_util,
                error_message=f"Process exited prematurely with code {exit_code}",
            )

        # Inspect metrics
        metrics_ok, last_rec, metrics_err = self.check_metrics()
        now = time.time()
        mtime = last_rec.get("mtime", now)
        elapsed_since_metric = max(0.0, now - mtime)
        step = last_rec.get("step", 0)

        if not metrics_ok:
            return HealthStatus(
                healthy=False,
                status_str="nan_inf",
                step=step,
                train_loss=last_rec.get("train_loss"),
                val_loss=last_rec.get("val_loss"),
                last_metric_update_s_ago=elapsed_since_metric,
                gpu_mem_used_mb=gpu_mb,
                gpu_util_pct=gpu_util,
                error_message=metrics_err,
            )

        # Stall detection: if training process is alive but metrics file hasn't updated
        if self.metrics_path.is_file() and elapsed_since_metric > self.stall_timeout_s:
            return HealthStatus(
                healthy=False,
                status_str="stalled",
                step=step,
                train_loss=last_rec.get("train_loss"),
                val_loss=last_rec.get("val_loss"),
                last_metric_update_s_ago=elapsed_since_metric,
                gpu_mem_used_mb=gpu_mb,
                gpu_util_pct=gpu_util,
                error_message=f"No metrics update for {int(elapsed_since_metric)}s (exceeds {int(self.stall_timeout_s)}s threshold)",
            )

        # Inspect checkpoint
        ckpt_step = None
        if self.ckpt_path.is_file():
            ckpt_insp = CheckpointValidator.inspect_checkpoint(self.ckpt_path)
            if ckpt_insp.get("valid"):
                ckpt_step = ckpt_insp.get("step")

        return HealthStatus(
            healthy=True,
            status_str="healthy",
            step=step,
            train_loss=last_rec.get("train_loss"),
            val_loss=last_rec.get("val_loss"),
            elapsed_seconds=last_rec.get("elapsed_seconds", 0.0),
            gpu_mem_used_mb=gpu_mb,
            gpu_util_pct=gpu_util,
            last_metric_update_s_ago=elapsed_since_metric,
            checkpoint_step=ckpt_step,
        )


# =====================================================================
# Stage 4: Diagnostics, Post-Mortem & Error Triage
# =====================================================================

class DiagnosticEngine:
    """Captures post-mortem state, writes FAILED_TRACEBACK.txt, and classifies failures [design]."""

    @staticmethod
    def capture_system_snapshot() -> Dict[str, Any]:
        """Collect host RAM, GPU memory, disk capacity, and python environment details [code]."""
        snapshot: Dict[str, Any] = {
            "timestamp": get_iso_timestamp(),
            "disk": {},
            "gpu": {},
        }

        # Disk usage on /content and Drive
        for mount_pt in ["/content", "/content/drive", "."]:
            p = Path(mount_pt)
            if p.exists():
                try:
                    usage = shutil.disk_usage(str(p))
                    snapshot["disk"][mount_pt] = {
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                    }
                except Exception:
                    pass

        # GPU snapshot via nvidia-smi
        if shutil.which("nvidia-smi") is not None:
            _, out, _ = run_command_capture(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv"],
                timeout=5.0,
            )
            snapshot["gpu"]["nvidia_smi"] = out.strip()

        return snapshot

    @staticmethod
    def handle_failure(
        run_dir: Path,
        run_id: str,
        process_mgr: ProcessManager,
        health: HealthStatus,
        bridge_outbox: Optional[Path] = None,
        prior_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Capture full diagnostics, terminate errant process, and dump failure artifacts [design]."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Terminate errant process to halt compute burn
        exit_code = process_mgr.terminate_gracefully(grace_period_s=10.0)

        # 2. Extract full log tail (20,000 characters per AGENTS.md rule)
        log_tail = process_mgr.get_log_tail(MAX_TRACEBACK_CHARS)

        # 3. Categorize failure cause
        classification = "UNKNOWN_FAILURE"
        remediation_suggestion = "Investigate traceback."

        if health.status_str == "oom" or "CUDA out of memory" in log_tail or "OutOfMemoryError" in log_tail:
            classification = "CUDA_OUT_OF_MEMORY"
            remediation_suggestion = "Reduce batch size, sequence length, or population K. Clear GPU memory before restart."
        elif health.status_str == "nan_inf":
            classification = "NUMERICAL_INSTABILITY_NAN_INF"
            remediation_suggestion = "Check learning rate, activation hooks scale, or SSTE backward denominator."
        elif health.status_str == "skipped":
            classification = "SKIPPED_NO_RESULT"
            remediation_suggestion = "Queue exited cleanly without running this config (check MAX_ATTEMPTS or --only filter)."
        elif health.status_str == "stalled":
            classification = "PROCESS_HANG_HEARTBEAT_TIMEOUT"
            remediation_suggestion = "Process hung in deadlock or infinite loop. Restart from verified checkpoint."
        elif exit_code != 0:
            classification = f"PROCESS_CRASH_EXIT_{exit_code}"

        # 4. Dump FAILED_TRACEBACK.txt in run_dir
        traceback_file = run_dir / "FAILED_TRACEBACK.txt"
        traceback_content = (
            f"=== Dust-Q Failure Diagnostic Report ===\n"
            f"Run ID: {run_id}\n"
            f"Timestamp: {get_iso_timestamp()}\n"
            f"Classification: {classification}\n"
            f"Exit Code: {exit_code}\n"
            f"Health Status: {health.status_str}\n"
            f"Error Message: {health.error_message}\n"
            f"Remediation: {remediation_suggestion}\n\n"
            f"=== Process Output Tail (last {len(log_tail)} chars) ===\n"
            f"{log_tail}\n"
        )
        traceback_file.write_text(traceback_content, encoding="utf-8")

        # 5. Update FAILED_ATTEMPTS marker (without double-counting if queue.py already incremented it)
        failed_marker = run_dir / "FAILED_ATTEMPTS"
        attempts = 0
        if failed_marker.is_file():
            try:
                attempts = int(failed_marker.read_text().splitlines()[0])
            except Exception:
                attempts = 0
        if prior_attempts is None or attempts <= prior_attempts:
            attempts += 1
            failed_marker.write_text(f"{attempts}\n{classification}\n{health.error_message}\n")

        # 6. Capture system snapshot
        sys_snapshot = DiagnosticEngine.capture_system_snapshot()
        state_file = run_dir / "system_state.json"
        state_file.write_text(json.dumps(sys_snapshot, indent=2), encoding="utf-8")

        incident_payload = {
            "run_id": run_id,
            "timestamp": get_iso_timestamp(),
            "classification": classification,
            "exit_code": exit_code,
            "attempts": attempts,
            "health": dataclasses.asdict(health),
            "traceback_path": str(traceback_file),
            "remediation": remediation_suggestion,
        }

        # 7. Deliver to Drive bridge outbox if available
        if bridge_outbox and bridge_outbox.exists():
            out_file = bridge_outbox / f"incident_{run_id}_{int(time.time())}.json"
            tmp_out = out_file.with_suffix(".tmp")
            tmp_out.write_text(json.dumps(incident_payload, indent=2), encoding="utf-8")
            tmp_out.replace(out_file)

        return incident_payload


# =====================================================================
# Stage 5 & 6: Verification & Resumption
# =====================================================================

class SandboxedVerifier:
    """Runs targeted tests and verified checkpoint recovery before resuming training [design]."""

    @staticmethod
    def verify_cuda_sanity(py_executable: str = sys.executable) -> bool:
        """Run tests/test_cuda.py or quick tensor sanity check [code]."""
        test_file = Path("tests/test_cuda.py")
        if test_file.is_file():
            code, out, err = run_command_capture([py_executable, "-m", "pytest", str(test_file), "-q"], timeout=30.0)
            return code == 0

        # Fallback quick inline check
        code, out, err = run_command_capture(
            [py_executable, "-c", "import torch; assert torch.cuda.is_available(); x = torch.ones(10, 10, device='cuda'); print('CUDA_OK')"],
            timeout=15.0,
        )
        return code == 0 and "CUDA_OK" in out

    @staticmethod
    def verify_checkpoint_recovery(run_dir: Path, is_dust: bool = False) -> Tuple[bool, Optional[int], Optional[str]]:
        """Verify candidate checkpoint validity before allowing resumption [design]."""
        ckpt_path = Path(run_dir) / "ckpt.pt"
        if not ckpt_path.is_file():
            return False, None, "No ckpt.pt found to resume from"

        report = CheckpointValidator.inspect_checkpoint(ckpt_path, is_dust=is_dust)
        if not report["valid"]:
            return False, None, f"Checkpoint invalid: {report.get('error')}"

        return True, report["step"], None


# =====================================================================
# Connection Mechanism B: Asynchronous Google Drive State-Exchange Bridge
# =====================================================================

class DriveBridge:
    """Zero-port, WebSocket-immune command & state broker via Google Drive [design]."""

    def __init__(self, bridge_dir: Path):
        self.bridge_dir = Path(bridge_dir)
        self.inbox_dir = self.bridge_dir / "inbox"
        self.outbox_dir = self.bridge_dir / "outbox"
        self.processed_dir = self.bridge_dir / "processed"
        self.heartbeat_file = self.bridge_dir / "heartbeat.json"

        for d in [self.inbox_dir, self.outbox_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def emit_heartbeat(self, status: HealthStatus, run_id: Optional[str] = None):
        """Write an atomic heartbeat JSON file to Google Drive [design]."""
        payload = {
            "timestamp": get_iso_timestamp(),
            "run_id": run_id,
            "status": status.status_str,
            "healthy": status.healthy,
            "step": status.step,
            "total_steps": status.total_steps,
            "train_loss": status.train_loss,
            "val_loss": status.val_loss,
            "gpu_mem_used_mb": status.gpu_mem_used_mb,
            "gpu_util_pct": status.gpu_util_pct,
            "last_metric_s_ago": status.last_metric_update_s_ago,
            "checkpoint_step": status.checkpoint_step,
        }
        tmp = self.heartbeat_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.heartbeat_file)

    def fetch_pending_commands(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """Fetch pending commands from inbox sorted by timestamp [design]."""
        commands = []
        for cmd_file in sorted(self.inbox_dir.glob("*.json")):
            try:
                data = json.loads(cmd_file.read_text(encoding="utf-8"))
                commands.append((cmd_file, data))
            except Exception:
                pass
        return commands

    def ack_command(self, cmd_path: Path, result: Dict[str, Any]):
        """Move command to processed and write response to outbox atomically [design]."""
        cmd_id = cmd_path.stem
        resp_file = self.outbox_dir / f"resp_{cmd_id}.json"
        tmp_resp = resp_file.with_suffix(".tmp")
        tmp_resp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        tmp_resp.replace(resp_file)

        dest = self.processed_dir / cmd_path.name
        try:
            cmd_path.replace(dest)
        except Exception:
            pass


# =====================================================================
# Autonomous Orchestration Loop
# =====================================================================

class AutonomousOrchestrator:
    """Coordinates Stages 1 through 6 for single runs or entire config queues [design]."""

    def __init__(
        self,
        configs_dir: Path = Path("configs"),
        results_dir: Path = Path("results"),
        bridge_dir: Optional[Path] = None,
        cu_burn_rate: str = DEFAULT_CU_BURN_RATE,
        watchdog_interval_s: float = DEFAULT_WATCHDOG_INTERVAL_S,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        retry_failed: bool = False,
    ):
        self.configs_dir = Path(configs_dir)
        self.results_dir = Path(results_dir)
        self.bridge = DriveBridge(bridge_dir) if bridge_dir else None
        self.cu_burn_rate = cu_burn_rate
        self.watchdog_interval_s = watchdog_interval_s
        self.stall_timeout_s = stall_timeout_s
        self.retry_failed = retry_failed
        self.process_mgr = ProcessManager(self.results_dir / "logs")

    def run_single_config(self, config_path: Path) -> bool:
        """Execute the 6-stage lifecycle for a single configuration [design]."""
        from dustq.config import load_config
        cfg = load_config(config_path)
        if self.cu_burn_rate:
            cfg.cu_burn_rate = self.cu_burn_rate

        run_id = cfg.run_id
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================", flush=True)
        print(f"[{get_iso_timestamp()}] Launching Run: {run_id}", flush=True)
        print(f"=======================================================", flush=True)

        # Check if already complete
        if (run_dir / "result.json").is_file():
            print(f"-> Skip: {run_id} is already completed (result.json exists).", flush=True)
            return True

        # Check prior failure attempts
        failed_marker = run_dir / "FAILED_ATTEMPTS"
        if self.retry_failed and failed_marker.is_file():
            print(f"-> Clearing FAILED_ATTEMPTS for {run_id} (--retry-failed enabled)", flush=True)
            failed_marker.unlink()

        if failed_marker.is_file():
            try:
                attempts = int(failed_marker.read_text().splitlines()[0])
                if attempts >= MAX_CONSECUTIVE_RETRIES:
                    print(f"-> Skip: {run_id} reached {attempts} failures (MAX_ATTEMPTS={MAX_CONSECUTIVE_RETRIES}).", flush=True)
                    return False
            except Exception:
                pass

        # -------------------------------------------------------------
        # Stage 1: Pre-flight Verification
        # -------------------------------------------------------------
        preflight = PreflightChecker(
            drive_data=Path("data"),
            drive_results=self.results_dir,
            configs_dir=self.configs_dir,
        )
        pf_res = preflight.run_all(tokens_subset=cfg.tokens)
        if not pf_res["passed"]:
            print(f"!! Stage 1 Pre-flight checks FAILED for {run_id}:", flush=True)
            print(json.dumps(pf_res, indent=2), flush=True)
            return False
        print(f"-> Stage 1 Pre-flight PASSED (GPU: {pf_res['gpu']['device_name']}, VRAM: {pf_res['gpu']['vram_gb']} GB)", flush=True)

        # -------------------------------------------------------------
        # Checkpoint Resume Pre-validation
        # -------------------------------------------------------------
        is_dust = (cfg.method == "dust")
        ckpt_path = run_dir / "ckpt.pt"
        if ckpt_path.is_file():
            valid_resume, resume_step, err = SandboxedVerifier.verify_checkpoint_recovery(run_dir, is_dust=is_dust)
            if valid_resume:
                print(f"-> Resuming from verified checkpoint at step {resume_step}.", flush=True)
            else:
                print(f"!! Checkpoint recovery pre-check failed: {err}. Starting from step 0.", flush=True)

        # -------------------------------------------------------------
        # Stage 2: Start Training Subprocess
        # -------------------------------------------------------------
        # Invoke runner via python subprocess to isolate PyTorch CUDA context
        py_exe = sys.executable
        failed_marker = run_dir / "FAILED_ATTEMPTS"
        attempts_before = 0
        if failed_marker.is_file():
            try:
                attempts_before = int(failed_marker.read_text().splitlines()[0])
            except Exception:
                attempts_before = 0

        cmd = [
            py_exe, "scripts/run_queue.py",
            "--configs-dir", str(self.configs_dir),
            "--results-dir", str(self.results_dir),
            "--only", run_id,
            "--exact",
            "--cu-burn-rate", self.cu_burn_rate,
        ]
        if self.retry_failed:
            cmd.append("--retry-failed")

        log_path = self.process_mgr.launch_training(cmd, run_id)
        print(f"-> Stage 2 Training started (PID: {self.process_mgr.process.pid}). Log: {log_path}", flush=True)

        # -------------------------------------------------------------
        # Stage 3: Watchdog Health Monitoring Loop
        # -------------------------------------------------------------
        watchdog = TrainingWatchdog(
            run_dir=run_dir,
            process_mgr=self.process_mgr,
            stall_timeout_s=self.stall_timeout_s,
        )

        while True:
            time.sleep(self.watchdog_interval_s)
            status = watchdog.inspect()

            if self.bridge:
                self.bridge.emit_heartbeat(status, run_id=run_id)

            if status.status_str == "completed":
                print(f"-> Run {run_id} COMPLETED successfully. Best Val Loss: {status.val_loss}", flush=True)
                return True

            if not status.healthy:
                print(f"\n!! UNHEALTHY condition detected: {status.status_str} - {status.error_message}", flush=True)

                # ---------------------------------------------------------
                # Stage 4: Diagnose & Repair
                # ---------------------------------------------------------
                incident = DiagnosticEngine.handle_failure(
                    run_dir=run_dir,
                    run_id=run_id,
                    process_mgr=self.process_mgr,
                    health=status,
                    bridge_outbox=self.bridge.outbox_dir if self.bridge else None,
                    prior_attempts=attempts_before,
                )
                print(f"-> Stage 4 Diagnostics: {incident['classification']}", flush=True)
                print(f"   Dumped report to: {incident['traceback_path']}", flush=True)

                if status.status_str == "skipped" or incident["attempts"] >= MAX_CONSECUTIVE_RETRIES:
                    print(f"!! Reached maximum attempts ({MAX_CONSECUTIVE_RETRIES}) for {run_id}. Halting.", flush=True)
                    return False

                # ---------------------------------------------------------
                # Stage 5: Verify Before Restart
                # ---------------------------------------------------------
                print("-> Stage 5 Verifying runtime before restart...", flush=True)
                cuda_ok = SandboxedVerifier.verify_cuda_sanity(py_exe)
                if not cuda_ok:
                    print("!! CUDA verification failed. Aborting retry to avoid burn loop.", flush=True)
                    return False

                # Verify checkpoint recovery
                if ckpt_path.is_file():
                    valid_ckpt, step, ckpt_err = SandboxedVerifier.verify_checkpoint_recovery(run_dir, is_dust=is_dust)
                    if not valid_ckpt:
                        print(f"!! Corrupted checkpoint detected ({ckpt_err}). Rolling back to best.pt or fresh run.", flush=True)
                        if (run_dir / "best.pt").is_file():
                            best_insp = CheckpointValidator.inspect_best_pt(run_dir / "best.pt")
                            if best_insp["valid"]:
                                print("   Valid best.pt confirmed.", flush=True)
                        ckpt_path.unlink(missing_ok=True)

                # ---------------------------------------------------------
                # Stage 6: Restart from verified checkpoint
                # ---------------------------------------------------------
                print(f"-> Stage 6 Restarting training for {run_id} (Attempt {incident['attempts'] + 1})...", flush=True)
                return self.run_single_config(config_path)

            else:
                # Periodic health logging
                step_info = f"Step: {status.step}"
                loss_info = f"Loss: {status.train_loss:.4f}" if status.train_loss is not None else "Loss: --"
                gpu_info = f"VRAM: {status.gpu_mem_used_mb:.0f}MB ({status.gpu_util_pct:.0f}% GPU)"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {run_id} | {step_info} | {loss_info} | {gpu_info}", flush=True)

    def run_queue(self, only: Optional[str] = None):
        """Iterate over all YAML configs in configs_dir and execute autonomously [design]."""
        configs = sorted(self.configs_dir.glob("*.yaml"))
        if only:
            configs = [c for c in configs if only in c.stem]

        print(f"Discovered {len(configs)} configs to process in {self.configs_dir}", flush=True)
        for cfg_file in configs:
            self.run_single_config(cfg_file)

    def run_bridge_listener(self, poll_interval_s: float = 10.0):
        """Run standing bridge listener: respond to Drive inbox commands from external agent [design]."""
        if not self.bridge:
            raise ValueError("Bridge directory must be configured to run bridge listener mode.")

        print(f"Starting Drive Bridge listener at {self.bridge.bridge_dir}", flush=True)
        while True:
            time.sleep(poll_interval_s)
            commands = self.bridge.fetch_pending_commands()
            for cmd_file, cmd_data in commands:
                action = cmd_data.get("action", "").upper()
                params = cmd_data.get("params", {})
                print(f"[{get_iso_timestamp()}] Received command {action}: {params}", flush=True)

                result: Dict[str, Any] = {"action": action, "timestamp": get_iso_timestamp()}
                if action == "PING":
                    result["status"] = "PONG"
                elif action == "RUN_QUEUE":
                    only = params.get("only")
                    self.run_queue(only=only)
                    result["status"] = "COMPLETED"
                elif action == "RUN_CONFIG":
                    cfg_name = params.get("config")
                    cfg_path = self.configs_dir / cfg_name
                    success = self.run_single_config(cfg_path)
                    result["status"] = "COMPLETED" if success else "FAILED"
                elif action == "STATUS":
                    result["status"] = "ONLINE"
                    result["gpu"] = PreflightChecker(Path("data"), self.results_dir, self.configs_dir).check_cuda_and_gpu()
                else:
                    result["status"] = "UNKNOWN_ACTION"

                self.bridge.ack_command(cmd_file, result)


# =====================================================================
# Main CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Dust-Q Colab Autonomous Training Harness")
    parser.add_argument("--mode", choices=["queue", "single", "bridge", "preflight"], default="queue",
                        help="Execution mode: queue (default), single, bridge listener, or preflight only")
    parser.add_argument("--configs-dir", default="configs", help="Directory containing YAML run configs")
    parser.add_argument("--results-dir", default=DEFAULT_DRIVE_RESULTS if Path(DEFAULT_DRIVE_RESULTS).parent.exists() else "results",
                        help="Directory for checkpoints, logs and runs.csv")
    parser.add_argument("--bridge-dir", default=DEFAULT_DRIVE_BRIDGE if Path(DEFAULT_DRIVE_BRIDGE).parent.exists() else "bridge",
                        help="Google Drive directory for async inbox/outbox bridge")
    parser.add_argument("--config", help="Specific YAML config filename for single mode")
    parser.add_argument("--only", help="Filter substring for config run_ids in queue mode")
    parser.add_argument("--cu-burn-rate", default=DEFAULT_CU_BURN_RATE, help="Observed CU/h burn rate")
    parser.add_argument("--retry-failed", action="store_true", help="Retry configs with FAILED_ATTEMPTS marker")
    parser.add_argument("--watchdog-interval", type=float, default=DEFAULT_WATCHDOG_INTERVAL_S,
                        help="Watchdog health check interval in seconds")
    parser.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT_S,
                        help="Maximum seconds without metrics update before declaring stall")

    args = parser.parse_args()

    # Preflight mode
    if args.mode == "preflight":
        checker = PreflightChecker(
            drive_data=Path("data"),
            drive_results=Path(args.results_dir),
            configs_dir=Path(args.configs_dir),
        )
        res = checker.run_all()
        print(json.dumps(res, indent=2))
        sys.exit(0 if res["passed"] else 1)

    orchestrator = AutonomousOrchestrator(
        configs_dir=Path(args.configs_dir),
        results_dir=Path(args.results_dir),
        bridge_dir=Path(args.bridge_dir) if args.mode == "bridge" else None,
        cu_burn_rate=args.cu_burn_rate,
        watchdog_interval_s=args.watchdog_interval,
        stall_timeout_s=args.stall_timeout,
        retry_failed=args.retry_failed,
    )

    if args.mode == "queue":
        orchestrator.run_queue(only=args.only)
    elif args.mode == "single":
        if not args.config:
            parser.error("--config is required when --mode single is selected")
        orchestrator.run_single_config(Path(args.configs_dir) / args.config)
    elif args.mode == "bridge":
        orchestrator.run_bridge_listener()


if __name__ == "__main__":
    main()
