# Review-fixes implementation plan (2026-10-14)

> **For agentic workers:** steps use checkbox syntax. This plan compresses the standard
> red-green cycle per task (every fix is a pre-verified reproduction from
> `docs/reports/REVIEW-2026-10-11.md` + the owner's second review; the spec at
> `docs/specs/2026-10-11-review-fixes-design.md` is the authority). Each task ends with a
> targeted pytest run and a commit.

**Goal:** fix B1–B5, C1–C3, C5, M1–M7 (+ review-2 revisions) so the first Colab L4 session is valid and productive.

**Architecture:** surgical edits to `dustq/` + `scripts/` + doc one-liners; three new/extended test files; no upstream changes.

**Tech stack:** PyTorch (CPU-verified, CUDA-gated tests), pytest, existing `dustq` package.

---

### Task 1 (B1): n_kv_head in SIZES + per-size forward test

**Files:** Modify `dustq/config.py` (SIZES), `tests/test_phase0.py` (add test)

```python
# dustq/config.py — replace SIZES with:
SIZES = {
    "L2d128": dict(n_layer=2, n_embd=128, n_head=2, n_kv_head=2),
    "L4d256": dict(n_layer=4, n_embd=256, n_head=4, n_kv_head=4),
    "L8d512": dict(n_layer=8, n_embd=512, n_head=8, n_kv_head=8),
    "L16d1024": dict(n_layer=16, n_embd=1024, n_head=16, n_kv_head=16),
}
```

```python
# tests/test_phase0.py — add:
@pytest.mark.parametrize("size", ["L2d128", "L4d256", "L8d512", "L16d1024"])
def test_size_forwards(size):
    from dustq.config import SIZES
    model = runner.build_model(SIZES[size], "fp", "cpu", seed=0)
    assert model.config.n_kv_head == model.config.n_head
    out = model(torch.randint(0, 4096, (2, 64)))
    assert out.shape == (2, 64, 4096) and torch.isfinite(out).all()
```

- [ ] Run `.venv\Scripts\python -m pytest tests/test_phase0.py -k size_forwards -q` → 4 passed. Commit: `fix(t1/B1): n_kv_head per size; per-size forward test`.

### Task 2 (B3): precision spellings

**Files:** Modify `dustq/qmodel.py` (parse_precision), `tests/test_phase0.py` (add test)

```python
# dustq/qmodel.py — replace the body of parse_precision:
def parse_precision(precision: str) -> tuple[str, int, str]:
    """'fp', 'w8a8', 'w1.58a8', 'w1.58a4', 'w1.58a2'; run-id spellings ('w1p58a4') accepted."""
    aliases = {"w1p58a4": "w1.58a4", "w1p58a8": "w1.58a8", "w1p58a2": "w1.58a2"}
    key = aliases.get(precision.lower(), precision.lower()).replace("_", "").replace(".", "")
    table = {"fp": ("fp", 0, "absmax"), "w8a8": ("int8", 8, "absmax"),
             "w158a8": ("ternary", 8, "absmax"), "w158a4": ("ternary", 4, "absmax"),
             "w158a2": ("ternary", 2, "absmax")}
    if key not in table:
        raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(table)} (dotted or run-id spelling)")
    return table[key]
```

```python
# tests/test_phase0.py — add:
def test_precision_spellings():
    assert qmodel.parse_precision("w1.58a4") == ("ternary", 4, "absmax")
    assert qmodel.parse_precision("w1p58a4") == ("ternary", 4, "absmax")
    assert qmodel.parse_precision("W1_58_A2") == ("ternary", 2, "absmax")
```

- [ ] Run `-k precision_spellings -q` → passed. Commit: `fix(t1/B3): parse_precision accepts run-id spellings`.

### Task 3 (B2): benchmark size aliases + default precisions

**Files:** Modify `scripts/benchmark.py`

```python
# after SIZES import — add:
SIZE_ALIASES = {"S": "L4d256", "M": "L8d512"}
# --sizes default stays "S,M"; in the real-scale cells loop:
cells = [(SIZE_ALIASES.get(s, s), k, p) for s in args.sizes.split(",") ...]
# --precisions default: "fp,w1.58a4"
# concurrent_test: resolve aliases before use
```

- [ ] Covered end-to-end by Task 12's benchmark test. Commit with Task 12.

### Task 4 (B4): QLinear device/dtype passthrough

**Files:** Modify `dustq/quant.py` (QLinear.__init__ + swap_linear), `tests/test_phase0.py` (add test)

```python
# QLinear.__init__ gains device/dtype and forwards them to nn.Linear:
def __init__(self, in_features: int, out_features: int, mode: str = "ternary",
             bias: bool = False, device=None, dtype=None):
    super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
    ...
# swap_linear passes them:
q = QLinear(module.in_features, module.out_features, mode=mode,
            bias=module.bias is not None, device=module.weight.device, dtype=module.weight.dtype)
```

```python
# tests/test_phase0.py — add:
def test_swap_preserves_dtype_and_device():
    src = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    q = swap_linear(src, "ternary")
    assert q.weight.dtype == src.weight.dtype and q.weight.device == src.weight.device
    assert q.effective_weight().dtype == src.weight.dtype
```

- [ ] Run `-k swap_preserves -q` → passed. Commit: `fix(t1/B4): QLinear keeps source device/dtype`.

### Task 5 (B5): inject cfg lr/momentum into the Dust optimizer

**Files:** Modify `dustq/train.py` (_build), `dustq/runner.py` (run_dust), `tests/test_t1.py` (add test)

```python
# dustq/train.py, dust path after settings = recipe(...):
import dataclasses  # at top
settings = dataclasses.replace(settings, lr=cfg.lr, momentum=cfg.momentum)
# dustq/runner.py run_dust: same replace after its recipe() call (values come from the recipe when unset)
```

```python
# tests/test_t1.py — add:
def test_dust_lr_override_reaches_optimizer(tmp_path):
    cfg = tiny_cfg(lr=0.001, momentum=0.5, output_dir=str(tmp_path))
    r = Runner(cfg); r._build()
    base = [g for g in r.optimizer.param_groups if g["lr"] not in (1000.0, 0.3, 0.03)]
    assert base and base[0]["lr"] == pytest.approx(0.001) and base[0]["momentum"] == pytest.approx(0.5)
```

- [ ] Run `-k lr_override -q` → passed. Commit: `fix(t1/B5): cfg lr/momentum reach the Dust optimizer`.

### Task 6 (C1): fp32 activation-quantizer math

**Files:** Modify `dustq/quant.py` (ste_act_quantize), `tests/test_phase0.py` (add test)

```python
# ste_act_quantize — first line of the body:
x = x.float()  # quantizer math is fp32 (Plan v3.1 4.2); return dtype stays fp32.
# (docstring updated to note the residual bf16 matmul-cast caveat until T5)
```

```python
# tests/test_phase0.py — add:
def test_act_quantizer_fp32_math():
    xb = (torch.randn(64, 128) * 3).bfloat16()
    out = ste_act_quantize(xb, 8)
    assert out.dtype == torch.float32
    assert torch.equal(out, ste_act_quantize(xb.float(), 8))
```

- [ ] Run `-k fp32_math -q` → passed. Commit: `fix(t2/C1): activation quantizer math in fp32`.

### Task 7 (C2): full-budget schedule for truncated runs

**Files:** Modify `dustq/train.py` (_load_data), `tests/test_t1.py` (add test)

```python
# _load_data, file mode — replace the load_sequences line:
default_steps = (dust.DEFAULT_CONFIGS.get(cfg.tokens, {}) or {}).get("steps") \
    or (BP_DEFAULTS.get(cfg.tokens, {}) or {}).get("steps")
rows_needed = max(cfg.steps, default_steps or cfg.steps) * 8
train = load_sequences(Path(cfg.data_dir) / f"train_{cfg.tokens}.pt", rows_needed)
# (import dust at module top already provides DEFAULT_CONFIGS; BP_DEFAULTS from .config)
```

```python
# tests/test_t1.py — add:
@pytest.mark.skipif(not Path("data/train_1M.pt").exists(), reason="data files not prepared")
def test_truncated_run_uses_full_budget_schedule(tmp_path):
    cfg = RunConfig(task="smoke", size="L4d256", tokens="1M", method="backprop", seed=0,
                    data_dir="data", steps=2, output_dir=str(tmp_path)).resolved()
    r = Runner(cfg); r._build(); r._load_data()
    train = load_sequences("data/train_1M.pt", 61 * 8)
    expected = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed + 1))]
    assert torch.equal(r.training, expected)   # 2-step run sees the 61-step run's batches
```

- [ ] Run `-k full_budget -q` → passed. Commit: `fix(t2/C2): truncated runs keep the full-budget schedule`.

### Task 8 (C3): self-describing atomic best.pt + load_best + reconcile

**Files:** Modify `dustq/train.py`, `tests/test_t1.py` (add test)

```python
# dustq/train.py:
def _save_best(self, best: float):
    payload = {"best": float(best),
               "state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}
    tmp = self.run_dir / "best.pt.tmp"
    torch.save(payload, tmp)
    tmp.replace(self.run_dir / "best.pt")

def load_best(path, map_location):
    """The only sanctioned reader of our best.pt ({"best", "state"}; not upstream's bare format)."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return payload["best"], payload["state"]

# train() resume path, after _load_ckpt:
best_path = self.run_dir / "best.pt"
if best_path.exists():
    try:
        best = min(best, load_best(best_path, self.device)[0])
    except Exception:
        pass  # legacy bare best.pt: keep the ckpt value
# _finalize: load via load_best; report best_val from the file (authoritative):
_, state = load_best(self.run_dir / "best.pt", self.device)
self.model.load_state_dict(state)
result = dict(best_val=load_best(...)[0] ...)
```

```python
# tests/test_t1.py — add:
def test_resume_does_not_corrupt_best(tmp_path):
    cfg_b = tiny_cfg(output_dir=str(tmp_path / "b"), steps=2)
    Runner(cfg_b).train()
    run_dir = tmp_path / "b" / cfg_b.run_id
    (run_dir / "result.json").unlink()
    from dustq.train import load_best
    best, state = load_best(run_dir / "best.pt", "cpu")
    crafted = {"best": 7.0, "state": state}           # a "best" no later eval can beat
    torch.save(crafted, run_dir / "best.pt.tmp"); (run_dir / "best.pt.tmp").replace(run_dir / "best.pt")
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    ckpt["best"] = 99.0                                # stale ckpt, as after a crash
    torch.save(ckpt, run_dir / "ckpt.pt")
    cfg_c = tiny_cfg(output_dir=str(tmp_path / "b"), steps=4)
    Runner(cfg_c).train()
    best_after, state_after = load_best(run_dir / "best.pt", "cpu")
    assert best_after == 7.0 and torch.equal(state_after["lm_head.weight"], state["lm_head.weight"])
    result = json.loads((run_dir / "result.json").read_text())
    assert result["best_val"] == 7.0
```

- [ ] Run `-k corrupt_best -q` → passed (fails without the reconcile). Commit: `fix(t2/C3): atomic self-describing best.pt; resume reconciles best`.

### Task 9 (C5): low-bit absmax guard + caller exemptions

**Files:** Modify `dustq/qmodel.py` (register_act_hooks), `dustq/train.py`, `dustq/runner.py`, `scripts/benchmark.py`, `tests/test_phase0.py` (add test)

```python
# qmodel.register_act_hooks signature + guard:
def register_act_hooks(model, bits: int, scheme: str = "absmax", allow_lowbit_absmax: bool = False):
    if scheme == "absmax" and bits <= 4 and not allow_lowbit_absmax:
        raise ValueError(f"{bits}-bit absmax activations are degenerate (Plan v3.1 4.2); "
                         "smoke/tests pass allow_lowbit_absmax=True; real A4/A2 cells wait for T5 alpha_b")
# callers: runner.run_dust/run_backprop -> allow_lowbit_absmax=True (smoke harness)
# train.Runner._build -> allow_lowbit_absmax=(self.cfg.task == "smoke")
# benchmark.make_job -> allow_lowbit_absmax=True (timing is scheme-independent)
```

```python
# tests/test_phase0.py — add:
def test_lowbit_absmax_guard():
    model = runner.build_model(precision="w1.58a4", seed=0)
    with pytest.raises(ValueError):
        qmodel.register_act_hooks(model, 4, "absmax")
    handles = qmodel.register_act_hooks(model, 4, "absmax", allow_lowbit_absmax=True)
    for h in handles: h.remove()
```

- [ ] Run `-k lowbit_absmax -q` → passed; existing smoke tests still pass. Commit: `fix(t2/C5): guard degenerate low-bit absmax outside smoke`.

### Task 10 (M1): failure markers + --retry-failed

**Files:** Modify `dustq/queue.py`, `scripts/run_queue.py`, `tests/test_t1.py` (add test)

```python
# dustq/queue.py:
FAILED_MARKER = "FAILED_ATTEMPTS"; MAX_ATTEMPTS = 2

def _attempts(run_dir: Path) -> int:
    p = run_dir / FAILED_MARKER
    return int(p.read_text().splitlines()[0] or 0) if p.exists() else 0

# run_queue(..., retry_failed: bool = False): after the complete-check:
if retry_failed and (run_dir / FAILED_MARKER).exists():
    (run_dir / FAILED_MARKER).unlink()
if _attempts(run_dir) >= MAX_ATTEMPTS:
    statuses.append(f"skip {cfg.run_id} (failed {MAX_ATTEMPTS}x)"); continue
# in the except block, before appending the row:
n = _attempts(run_dir) + 1
(run_dir / FAILED_MARKER).write_text(f"{n}\n{note}\n")
# scripts/run_queue.py: add --retry-failed flag -> run_queue(..., retry_failed=args.retry_failed)
```

```python
# tests/test_t1.py — add:
def test_queue_failure_marker(tmp_path):
    import yaml
    configs = tmp_path / "configs"; configs.mkdir()
    cfg = tiny_cfg(method="backprop", lr=1e-3, momentum=0.9, data_dir=str(tmp_path / "missing"),
                   steps=1, output_dir="unused")
    (configs / "bad.yaml").write_text(yaml.safe_dump(cfg.to_yaml_dict()))
    results = str(tmp_path / "results")
    assert all(s.startswith("FAILED") for s in run_queue(configs, results))
    assert all(s.startswith("FAILED") for s in run_queue(configs, results))
    assert all(s.startswith("skip") for s in run_queue(configs, results))
    assert all(s.startswith("FAILED") for s in run_queue(configs, results, retry_failed=True))
    lines = (tmp_path / "results" / "runs.csv").read_text().splitlines()
    assert len(lines) == 5 and sum(1 for l in lines if ",failed," in l) == 3
```

- [ ] Run `-k failure_marker -q` → passed. Commit: `fix(t3/M1): failed-attempt markers with --retry-failed`.

### Task 11 (M2): truncate metrics.jsonl on resume

**Files:** Modify `dustq/train.py` (resume path), `tests/test_t1.py` (extend)

```python
# after _load_ckpt in train():
if history_path.exists():
    kept = [l for l in history_path.read_text().splitlines()
            if l.strip() and json.loads(l).get("step", 0) <= start]
    history_path.write_text("\n".join(kept) + ("\n" if kept else ""))
# tests/test_t1.py, in test_resume_exactness_dust after run_c.train():
steps_seen = [json.loads(l)["step"] for l in (tmp_path / "b" / cfg_c.run_id / "metrics.jsonl").read_text().splitlines() if l.strip()]
assert steps_seen == sorted(set(steps_seen)) == [0, 1, 2, 3, 4]
```

- [ ] Run `-k resume_exactness_dust -q` → passed. Commit: `fix(t3/M2): no duplicate metrics lines after resume`.

### Task 12 (M3): subprocess worker mode + 2-concurrent

**Files:** Modify `scripts/benchmark.py`, `tests/test_t1.py` (add tests)

```python
# benchmark.py: --worker flag runs one job and prints JSON: {"s_per_step": ...}
# worker args: --worker-size --worker-population --worker-precision --worker-steps --worker-scale
# concurrent_test spawns len(jobs) subprocesses [sys.executable, __file__, "--worker", ...],
# waits on all, returns wall (replaces the threaded version).
# --skip-concurrent flag for fast test paths.
```

```python
# tests/test_t1.py — add (also covers B2/B3 end-to-end, spec test #2):
def test_benchmark_end_to_end_tiny():
    import subprocess, sys as _sys
    r = subprocess.run([_sys.executable, "scripts/benchmark.py", "--device", "cpu", "--scale", "tiny",
                        "--sizes", "S", "--populations", "256", "--precisions", "fp,w1.58a4",
                        "--steps", "1", "--skip-concurrent"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert Path("results/benchmark_cpu.md").exists()
```

- [ ] Run `-k benchmark_end_to_end -q` → passed. Commit: `fix(t3/M3): subprocess 2-concurrent measurement + worker mode; benchmark e2e test`.

### Task 13 (M4): Colab sanity composition

**Files:** Create `tests/test_cuda.py`, Modify `scripts/colab_t1.py`

```python
# tests/test_cuda.py:
import math, pytest, torch
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
from dustq import runner

@pytest.mark.parametrize("precision", ["fp", "w1.58a4"])
def test_one_dust_step_on_gpu(precision):
    assert all(math.isfinite(l) for l in runner.run_dust(precision=precision, steps=1, device="cuda"))

@pytest.mark.parametrize("precision", ["fp", "w1.58a4"])
def test_one_bp_step_on_gpu(precision):
    assert all(math.isfinite(l) for l in runner.run_backprop(precision=precision, steps=1, device="cuda"))
# colab_t1.py: t0 at cell start; sanity = test_phase0.py (CPU integrity) +
# test_t1.py -k resume (Runner picks CUDA) + tests/test_cuda.py (the B4-catcher).
```

- [ ] CPU: `-k cuda -q` → all skipped (expected, no local GPU). Commit: `fix(t3/M4): Colab sanity = integrity + CUDA-path + per-precision GPU steps`.

### Task 14 (review-2 claim 3): device-aware assertions in test_t1

**Files:** Modify `tests/test_t1.py`

```python
# assert_same_state + history comparisons become device-aware:
def _same(a, b, device):
    return torch.equal(a, b) if device.type == "cpu" else torch.allclose(a, b, rtol=1e-2, atol=1e-2)
# histories: steps exact; losses exact on CPU, allclose(rtol=1e-3, atol=1e-3) on CUDA.
```

- [ ] Run `-k resume -q` on CPU → bitwise path unchanged, passed. Commit: `fix(t3): device-aware resume assertions (bitwise CPU, tolerance CUDA)`.

### Task 15 (M5/M6/M7): doc one-liners + format-deviations note

**Files:** `docs/REPO_MAP.md` (relabel targets + our-format notes), `docs/HYPOTHESES.md` (n=5 unconditional), `docs/reports/T1.md` (addendum), `docs/OPEN_QUESTIONS.md` (log the pass)

- REPO_MAP reproduction-targets header: M values = Appendix C; S/10M 5.362 = main-post Figure 4 (size-study protocol). New "Our format deviations" note: `best.pt` is `{"best", "state"}` (read via `dustq.train.load_best`); `scripts/run_queue.py` naming.
- HYPOTHESES: "n = 5 seeds on the primary cell and its fp counterpart, unconditional (D2 approved; no optional stopping)."
- T1.md addendum: review pass B1–B5/C1–C3/C5/M1–M7 fixed pre-Colab; pointer to spec + plan.

- [ ] Commit: `docs: M5/M6/M7 review fixes`.

### Task 16: full verification (verifying-before-completion)

- [ ] `.venv\Scripts\python -m pytest tests -q` → all green (CUDA tests skip locally).
- [ ] `.venv\Scripts\python scripts/benchmark.py --device cpu --scale tiny --sizes S --populations 256,1024 --precisions fp,w1.58a4 --steps 1 --skip-concurrent` → exit 0, `results/benchmark_cpu.md` written.
- [ ] `git archive -o dustq.zip HEAD` regenerated.
- [ ] Full-suite + benchmark outputs recorded in the commit message; report to owner with Colab go/no-go + steps.

