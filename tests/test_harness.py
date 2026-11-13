"""Unit tests for the Colab Autonomous Harness (scripts/colab_autonomous_harness.py).

Tests Preflight, CheckpointValidator, TrainingWatchdog, DiagnosticEngine, and DriveBridge
using ephemeral temporary directories and synthetic mock objects.
"""

import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.colab_autonomous_harness import (
    CheckpointValidator,
    DiagnosticEngine,
    DriveBridge,
    HealthStatus,
    PreflightChecker,
    ProcessManager,
    SandboxedVerifier,
    TrainingWatchdog,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# =====================================================================
# 1. CheckpointValidator Tests
# =====================================================================

def test_checkpoint_validator_valid_payload(temp_dir):
    ckpt_path = temp_dir / "ckpt.pt"
    payload = {
        "model": {"weight": torch.randn(4, 4), "bias": torch.zeros(4)},
        "optimizer": {"state": {}, "param_groups": []},
        "step": 42,
        "best": 5.432,
        "history": [{"step": 42, "val_loss": 5.432}],
        "wall_prior": 120.5,
    }
    torch.save(payload, ckpt_path)

    report = CheckpointValidator.inspect_checkpoint(ckpt_path, is_dust=False)
    assert report["valid"] is True
    assert report["step"] == 42
    assert report["best"] == pytest.approx(5.432)
    assert report["num_params"] == 20
    assert len(report["non_finite_tensors"]) == 0


def test_checkpoint_validator_nan_weights(temp_dir):
    ckpt_path = temp_dir / "ckpt_nan.pt"
    bad_tensor = torch.tensor([1.0, float("nan"), 3.0])
    payload = {
        "model": {"bad_weight": bad_tensor},
        "optimizer": {},
        "step": 10,
        "best": 6.0,
        "history": [],
        "wall_prior": 0.0,
    }
    torch.save(payload, ckpt_path)

    report = CheckpointValidator.inspect_checkpoint(ckpt_path)
    assert report["valid"] is False
    assert "bad_weight" in report["non_finite_tensors"]
    assert "NaN/Inf weights detected" in report["error"]


def test_checkpoint_validator_missing_keys(temp_dir):
    ckpt_path = temp_dir / "ckpt_incomplete.pt"
    payload = {"model": {"w": torch.ones(2)}}  # missing optimizer, step, etc.
    torch.save(payload, ckpt_path)

    report = CheckpointValidator.inspect_checkpoint(ckpt_path)
    assert report["valid"] is False
    assert "missing_keys" in report
    assert "optimizer" in report["missing_keys"]


def test_checkpoint_validator_dust_generator_requirement(temp_dir):
    ckpt_path = temp_dir / "ckpt_dust.pt"
    payload = {
        "model": {"w": torch.ones(2)},
        "optimizer": {},
        "step": 5,
        "best": 4.0,
        "history": [],
        "wall_prior": 10.0,
        # missing 'generator' key
    }
    torch.save(payload, ckpt_path)

    # is_dust=False passes
    report_bp = CheckpointValidator.inspect_checkpoint(ckpt_path, is_dust=False)
    assert report_bp["valid"] is True

    # is_dust=True fails because generator is required for exact noise stream resumption
    report_dust = CheckpointValidator.inspect_checkpoint(ckpt_path, is_dust=True)
    assert report_dust["valid"] is False
    assert any("generator" in k for k in report_dust["missing_keys"])


def test_best_pt_validator(temp_dir):
    best_path = temp_dir / "best.pt"
    payload = {"best": 4.125, "state": {"w": torch.ones(2, 2)}}
    torch.save(payload, best_path)

    report = CheckpointValidator.inspect_best_pt(best_path)
    assert report["valid"] is True
    assert report["best"] == pytest.approx(4.125)


# =====================================================================
# 2. Watchdog & Metrics Parsing Tests
# =====================================================================

class MockProcessManager:
    def __init__(self, alive=True, exit_code=None, log_tail=""):
        self._alive = alive
        self._exit_code = exit_code
        self._log_tail = log_tail

    def is_alive(self):
        return self._alive

    def get_exit_code(self):
        return self._exit_code

    def get_log_tail(self, max_chars=20000):
        return self._log_tail


def test_watchdog_healthy_metrics(temp_dir):
    run_dir = temp_dir / "run_test"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.jsonl"

    metrics = [
        {"step": 1, "train_loss": 6.5, "val_loss": 6.4, "elapsed_seconds": 1.0},
        {"step": 2, "train_loss": 6.2, "val_loss": 6.1, "elapsed_seconds": 2.5},
    ]
    with metrics_path.open("w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

    proc_mgr = MockProcessManager(alive=True)
    watchdog = TrainingWatchdog(run_dir=run_dir, process_mgr=proc_mgr, stall_timeout_s=60.0)

    status = watchdog.inspect()
    assert status.healthy is True
    assert status.status_str == "healthy"
    assert status.step == 2
    assert status.train_loss == pytest.approx(6.2)
    assert status.val_loss == pytest.approx(6.1)


def test_watchdog_detects_nan_loss(temp_dir):
    run_dir = temp_dir / "run_nan"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.jsonl"

    with metrics_path.open("w") as f:
        f.write(json.dumps({"step": 1, "train_loss": 5.0}) + "\n")
        f.write('{"step": 2, "train_loss": NaN}\n')

    proc_mgr = MockProcessManager(alive=True)
    watchdog = TrainingWatchdog(run_dir=run_dir, process_mgr=proc_mgr)

    status = watchdog.inspect()
    assert status.healthy is False
    assert status.status_str == "nan_inf"
    assert "Non-finite" in status.error_message


def test_watchdog_detects_stall(temp_dir):
    run_dir = temp_dir / "run_stall"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.jsonl"

    with metrics_path.open("w") as f:
        f.write(json.dumps({"step": 10, "train_loss": 4.5}) + "\n")

    # Set file mtime to 400 seconds ago
    old_time = time.time() - 400
    os.utime(metrics_path, (old_time, old_time))

    proc_mgr = MockProcessManager(alive=True)
    watchdog = TrainingWatchdog(run_dir=run_dir, process_mgr=proc_mgr, stall_timeout_s=300.0)

    status = watchdog.inspect()
    assert status.healthy is False
    assert status.status_str == "stalled"
    assert "No metrics update" in status.error_message


def test_watchdog_detects_process_crash(temp_dir):
    run_dir = temp_dir / "run_crash"
    run_dir.mkdir()

    proc_mgr = MockProcessManager(alive=False, exit_code=1, log_tail="Traceback (most recent call last):\nRuntimeError: test failure")
    watchdog = TrainingWatchdog(run_dir=run_dir, process_mgr=proc_mgr)

    status = watchdog.inspect()
    assert status.healthy is False
    assert status.status_str == "crashed"
    assert "exited prematurely with code 1" in status.error_message


def test_watchdog_detects_oom(temp_dir):
    run_dir = temp_dir / "run_oom"
    run_dir.mkdir()

    oom_log = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    proc_mgr = MockProcessManager(alive=False, exit_code=1, log_tail=oom_log)
    watchdog = TrainingWatchdog(run_dir=run_dir, process_mgr=proc_mgr)

    status = watchdog.inspect()
    assert status.healthy is False
    assert status.status_str == "oom"


# =====================================================================
# 3. DiagnosticEngine Tests
# =====================================================================

class ControllableProcessManager:
    def __init__(self, log_tail=""):
        self._log_tail = log_tail
        self.terminated = False

    def is_alive(self):
        return not self.terminated

    def get_exit_code(self):
        return -15 if self.terminated else 0

    def terminate_gracefully(self, grace_period_s=1.0):
        self.terminated = True
        return -15

    def get_log_tail(self, max_chars=20000):
        return self._log_tail


def test_diagnostic_engine_dumps_traceback_and_state(temp_dir):
    run_dir = temp_dir / "run_diag"
    outbox = temp_dir / "bridge_outbox"
    outbox.mkdir(parents=True)

    proc_mgr = ControllableProcessManager(log_tail="Simulated CUDA fatal error at step 50")
    health = HealthStatus(
        healthy=False,
        status_str="stalled",
        step=50,
        error_message="Heartbeat timeout exceeded",
    )

    incident = DiagnosticEngine.handle_failure(
        run_dir=run_dir,
        run_id="test_run_001",
        process_mgr=proc_mgr,
        health=health,
        bridge_outbox=outbox,
    )

    assert incident["classification"] == "PROCESS_HANG_HEARTBEAT_TIMEOUT"
    assert (run_dir / "FAILED_TRACEBACK.txt").is_file()
    assert (run_dir / "FAILED_ATTEMPTS").is_file()
    assert (run_dir / "system_state.json").is_file()

    # Check FAILED_ATTEMPTS content
    attempts_content = (run_dir / "FAILED_ATTEMPTS").read_text()
    assert attempts_content.startswith("1\n")

    # Check incident delivered to bridge outbox
    outbox_files = list(outbox.glob("incident_test_run_001_*.json"))
    assert len(outbox_files) == 1


# =====================================================================
# 4. DriveBridge Protocol Tests
# =====================================================================

def test_drive_bridge_protocol(temp_dir):
    bridge_dir = temp_dir / "bridge"
    bridge = DriveBridge(bridge_dir)

    # 1. Test heartbeat emission
    status = HealthStatus(
        healthy=True,
        status_str="healthy",
        step=100,
        total_steps=1000,
        train_loss=4.12,
        val_loss=4.15,
        gpu_mem_used_mb=4500,
        gpu_util_pct=85,
    )
    bridge.emit_heartbeat(status, run_id="run_abc")

    heartbeat_path = bridge_dir / "heartbeat.json"
    assert heartbeat_path.is_file()
    hb_data = json.loads(heartbeat_path.read_text())
    assert hb_data["run_id"] == "run_abc"
    assert hb_data["step"] == 100
    assert hb_data["train_loss"] == pytest.approx(4.12)

    # 2. Test command processing (Inbox -> Ack/Outbox -> Processed)
    cmd_file = bridge.inbox_dir / "cmd_001.json"
    cmd_data = {"action": "RUN_CONFIG", "params": {"config": "test.yaml"}}
    cmd_file.write_text(json.dumps(cmd_data))

    pending = bridge.fetch_pending_commands()
    assert len(pending) == 1
    assert pending[0][1]["action"] == "RUN_CONFIG"

    # Ack command
    bridge.ack_command(pending[0][0], {"status": "SUCCESS", "details": "Done"})

    # Inbox should now be empty
    assert len(bridge.fetch_pending_commands()) == 0
    # Processed should contain cmd_001.json
    assert (bridge.processed_dir / "cmd_001.json").is_file()
    # Outbox should contain resp_cmd_001.json
    assert (bridge.outbox_dir / "resp_cmd_001.json").is_file()
