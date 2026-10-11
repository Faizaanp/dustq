"""Resumable training entry point (Plan v3.1 section 4.4): one Runner for every method.

Resume is exact: the only stochastic element in a Dust step is the estimator's device
generator (saved via DUSTQ.state_dict), and backprop has no training-time RNG at all;
data is a deterministic shuffled-prefix schedule. So a resumed run continues the same
noise stream, and tests/test_t1.py checks the final model matches an uninterrupted run
bitwise.

Evaluation mirrors upstream dust.evaluate but respects dtype ("auto" = bf16 on CUDA,
fp32 on CPU; "fp32" forces fp32 on CUDA too), fixing upstream's hard-coded bf16 eval
(REPO_MAP note) for the G1 dtype comparison.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch

from . import qmodel
from .backprop_ext import make_bp_optimizer, train_step
from .config import RunConfig
from .dust_ext import DUSTQ
from .recipes import recipe
from .runner import build_model, tiny_batch
from .upstream import ensure_upstream_on_path

ensure_upstream_on_path()

from dust import load_sequences  # noqa: E402
from dust import make_optimizer as dust_make_optimizer  # noqa: E402


def evaluate(model, rows, device, dtype: str = "auto", batch_size: int = 32) -> float:
    total, tokens = 0.0, 0
    use_bf16 = device.type == "cuda" and dtype != "fp32"
    for batch in rows.split(batch_size):
        batch = batch.to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        total += loss.item() * y.numel()
        tokens += y.numel()
    return total / tokens


def _force_fp32(model, estimator):
    """G1 helper: cast the bf16 embeddings back and switch the estimator autocast to fp32."""
    model.to(dtype=torch.float32)
    estimator.dtype = torch.float32


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class Runner:
    def __init__(self, cfg: RunConfig, device: str | torch.device | None = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run_dir = Path(cfg.output_dir) / cfg.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.run_dir / "ckpt.pt"
        self.model = self.estimator = self.optimizer = None

    # ---------- setup ----------

    def _build(self):
        cfg = self.cfg
        self.model = build_model(cfg.model_kwargs(), cfg.precision, self.device, cfg.seed)
        if cfg.method == "dust":
            self.model.requires_grad_(False)
            settings = recipe(cfg.tokens, cfg.population, self.model.config.n_layer)
            self.estimator = DUSTQ(self.model, settings, seed=cfg.seed, sigmas=cfg.sigmas)
            if cfg.dtype == "fp32" and self.device.type == "cuda":
                _force_fp32(self.model, self.estimator)
            if cfg.precision != "fp":  # AFTER estimator: hook order noise -> quantize
                _, bits, scheme = qmodel.parse_precision(cfg.precision)
                qmodel.register_act_hooks(self.model, bits, scheme)
            self.optimizer = dust_make_optimizer(self.model, settings)
        else:
            self.model.requires_grad_(True)
            if cfg.precision != "fp":
                _, bits, scheme = qmodel.parse_precision(cfg.precision)
                qmodel.register_act_hooks(self.model, bits, scheme)
            self.optimizer = make_bp_optimizer(self.model, cfg.lr, cfg.momentum)

    def _load_data(self):
        cfg = self.cfg
        if cfg.data_dir == "synthetic":
            vocab = self.model.config.vocab_size
            length = self.model.config.sequence_len
            x, y = tiny_batch(self.device, seed=cfg.seed + 1, sequences=8, length=length, vocab=vocab)
            self.training, self.targets = x, y
            self.val_rows = tiny_batch(torch.device("cpu"), seed=90 + cfg.seed, sequences=32,
                                       length=length, vocab=vocab)[0]
            self.test_rows = tiny_batch(torch.device("cpu"), seed=190 + cfg.seed, sequences=32,
                                        length=length, vocab=vocab)[0]
        else:
            train = load_sequences(Path(cfg.data_dir) / f"train_{cfg.tokens}.pt", cfg.steps * 8)
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed + 1))
            self.training = train[order]
            self.val_rows = load_sequences(Path(cfg.data_dir) / "val.pt", 544)
            self.test_rows = load_sequences(Path(cfg.data_dir) / "test.pt", 544)

    def _batch(self, step: int):
        if self.cfg.data_dir == "synthetic":
            return self.training, self.targets
        rows = self.training[(step - 1) * 8: step * 8].to(self.device)
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    # ---------- checkpointing ----------

    def _save_ckpt(self, step: int, best: float, history: list):
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step, "best": best, "history": history,
        }
        if self.estimator is not None:
            payload["generator"] = self.estimator.state_dict()
        tmp = self.ckpt_path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.ckpt_path)

    def _load_ckpt(self):
        payload = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.estimator is not None and "generator" in payload:
            self.estimator.load_state_dict(payload["generator"])
        return payload["step"], payload["best"], payload["history"]

    # ---------- training ----------

    def train(self) -> Path:
        cfg = self.cfg
        self._build()
        self._load_data()
        history_path = self.run_dir / "metrics.jsonl"
        start, best, history = 0, None, []

        if self.ckpt_path.exists():
            start, best, history = self._load_ckpt()
            if start >= cfg.steps:
                return self._finalize(best, history)
        else:
            (self.run_dir / "config.json").write_text(
                json.dumps({**cfg.to_yaml_dict(), "git_commit": _git_commit(),
                            "torch": torch.__version__,
                            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu"},
                           indent=2) + "\n")
            best = evaluate(self.model, self.val_rows, self.device, cfg.dtype)
            self._save_best()
            history = [dict(step=0, val_loss=best)]

        t_start, t_ckpt = time.monotonic(), time.monotonic()
        for step in range(start + 1, cfg.steps + 1):
            x, y = self._batch(step)
            if cfg.method == "dust":
                loss = self.estimator.step(x, y, self.optimizer)
            else:
                loss = train_step(self.model, self.optimizer, x, y)
            record = dict(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - t_start)
            if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
                value = evaluate(self.model, self.val_rows, self.device, cfg.dtype)
                record["val_loss"] = value
                if best is None or value < best:
                    best, improved = value, True
                    self._save_best()
                history.append(record)
            if step == cfg.steps or cfg.checkpoint_minutes <= 0 or \
                    (time.monotonic() - t_ckpt) >= cfg.checkpoint_minutes * 60:
                self._save_ckpt(step, best, history)
                t_ckpt = time.monotonic()
            with history_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

        if self.estimator is not None:
            self.estimator.close()
        return self._finalize(best, history)

    def _save_best(self):
        torch.save({k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                   self.run_dir / "best.pt")

    def _finalize(self, best, history) -> Path:
        model = self.model
        model.load_state_dict(torch.load(self.run_dir / "best.pt", map_location=self.device,
                                         weights_only=True))
        result = dict(
            best_val=best,
            test_at_best_val=evaluate(model, self.test_rows, self.device, self.cfg.dtype),
            history=history, steps=self.cfg.steps,
            wall_s=history[-1].get("elapsed_seconds", 0.0) if history else 0.0,
            params=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
            dtype="fp32" if (self.cfg.dtype == "fp32" or self.device.type == "cpu") else "bf16",
            commit=_git_commit(),
        )
        path = self.run_dir / "result.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        return path
