"""Run configs (one YAML per run, Plan v3.1 section 4 / v1 section 6) and the runs.csv schema.

Learning rates and momenta default to upstream's tuned values for the requested
(tokens, population) cell unless explicitly overridden — the resolved values are what
the runner logs, never the YAML defaults. Smoke runs (synthetic data) set their own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .upstream import ensure_upstream_on_path

ensure_upstream_on_path()

import dust  # noqa: E402

try:  # backprop defaults live in baselines/
    import baselines.backprop as _bp  # type: ignore
    BP_DEFAULTS = _bp.DEFAULT_CONFIGS
except Exception:  # vendored path layout differences; fall back to constants from the code read
    BP_DEFAULTS = {"1M": dict(lr=0.175, momentum=0.98, steps=61, eval_interval=2),
                   "10M": dict(lr=0.2, momentum=0.95, steps=610, eval_interval=10)}

SIZES = {
    "L2d128": dict(n_layer=2, n_embd=128, n_head=2, n_kv_head=2),
    "L4d256": dict(n_layer=4, n_embd=256, n_head=4, n_kv_head=4),
    "L8d512": dict(n_layer=8, n_embd=512, n_head=8, n_kv_head=8),
    "L16d1024": dict(n_layer=16, n_embd=1024, n_head=16, n_kv_head=16),
}

CSV_COLUMNS = [
    "run_id", "task", "model", "params", "tokens", "precision", "method", "K", "seed",
    "lr", "momentum", "noise_scales", "val_loss_best", "test_loss_at_best_val", "steps",
    "wall_s", "gpu", "dtype", "cu_burn_rate", "commit", "status", "notes",
]


def sanitize(name: str) -> str:
    return name.replace(".", "p").replace("/", "-")


@dataclass
class RunConfig:
    task: str = "smoke"
    size: str = "L4d256"                 # key of SIZES, or "tiny" for the CPU acceptance model
    tokens: str = "1M"                   # 1M | 10M | 20M (20M needs dustq/data.py)
    precision: str = "fp"                # fp | w8a8 | w1.58a8 | w1.58a4 | w1.58a2
    method: str = "dust"                # dust | backprop
    population: int = 256               # ignored (K0) for backprop
    seed: int = 0
    lr: float | None = None             # None -> upstream tuned default for the cell
    momentum: float | None = None
    sigmas: dict | None = None          # None -> DUSTQ defaults (upstream literals)
    ste_mode: str = "identity"          # identity = BP-STE; smoothed = BP-SSTE (T5)
    dtype: str = "auto"                 # auto = bf16 on CUDA, fp32 on CPU; or "fp32"
    data_dir: str = "data"              # where train_/val/test .pt live; "synthetic" for smoke
    output_dir: str = "results"
    steps: int | None = None            # None -> upstream default for tokens budget
    eval_interval: int | None = None
    checkpoint_minutes: float = 10.0
    notes: str = ""
    cu_burn_rate: str = ""              # filled from the Colab Resources tab by hand
    meta: dict = field(default_factory=dict)

    def resolved(self) -> "RunConfig":
        cfg = RunConfig(**asdict(self))
        if cfg.method == "dust":
            cell = dust.DEFAULT_CONFIGS[cfg.tokens]["optimizer"][cfg.population]
            defaults_steps = dust.DEFAULT_CONFIGS[cfg.tokens]["steps"]
            defaults_eval = dust.DEFAULT_CONFIGS[cfg.tokens]["eval_interval"]
        else:
            cell = BP_DEFAULTS[cfg.tokens]
            defaults_steps, defaults_eval = cell["steps"], cell["eval_interval"]
        cfg.lr = cfg.lr if cfg.lr is not None else cell["lr"]
        cfg.momentum = cfg.momentum if cfg.momentum is not None else cell["momentum"]
        cfg.steps = cfg.steps if cfg.steps is not None else defaults_steps
        cfg.eval_interval = cfg.eval_interval if cfg.eval_interval is not None else defaults_eval
        return cfg

    @property
    def run_id(self) -> str:
        k = self.population if self.method == "dust" else 0
        return f"{self.size}_{self.tokens}_{sanitize(self.precision)}_{self.method}_K{k}_s{self.seed}"

    def model_kwargs(self) -> dict:
        if self.size == "tiny":
            from .runner import TINY
            return dict(TINY)
        return dict(SIZES[self.size])

    def to_yaml_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        return d


def load_config(path: str | Path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw.pop("run_id", None)  # informational only; derived from the fields
    return RunConfig(**raw).resolved()


def load_configs_dir(directory: str | Path) -> list[tuple[str, RunConfig]]:
    out = []
    for path in sorted(Path(directory).glob("*.yaml")):
        out.append((path.name, load_config(path)))
    return out
