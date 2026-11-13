# AGENTS.md — Dust-Q

Read this before touching the repo. It is the contract for any agent (human or AI) working
here. Internal document: kept on disk, not shipped in the public repo.

## What this is

Dust-Q extends **Dust** (Q Labs' backprop-free pretraining via activation perturbation) to
**quantized transformers (ternary/int8 weights, A8/A4/A2 activations) trained from scratch**,
targeting arXiv then TMLR. Plan: `docs/planning/Dust-Q Implementation Plan v3.md` (v3.1).

## The non-negotiable rules

1. **Never invent numbers.** Tag every claim: `[paper]` (verified primary source),
   `[code]` (measured), `[design]` (a choice), `[unverified]` (not yet checked).
2. **No test-set tuning.** Selection by validation loss only.
3. **Log every run, including failures.** Per-task report in `docs/reports/`.
4. **Upstream is vendored unmodified** in `third_party/dust/`, pinned at
   `b20f7c03eac630b6441ba6f254128c1761dc45ad`. Never edit it.
5. **No model execution on the owner's machine.** Local runs are the test suite and smoke
   steps only; training, calibration, benchmarking and probing happen on Colab L4.
6. **Corrections sit next to the original.** When a claim is later shown wrong, the original
   line stays and a dated correction sits next to it. The audit trail is the value.

## The traps (each has cost us real time)

- **Stale zips.** `dustq.zip` is `git archive -o dustq.zip HEAD`. Rebuild after every
  commit. A stale zip produces phantom bugs that look like science.
- **PowerShell quoting.** Commit messages must go through a temp file:
  `git commit -F <file>`. Inline quotes corrupt multi-file commits. Always use `git commit -F`.
- **Swallowed errors.** Any subprocess that matters gets captured; tails printed on failure.
  Write data before pretty-printing it.
- **`git rm --cached` alone never sticks.** Add the `.gitignore` rule first, then `git add -A`,
  then verify the file is gone from `git ls-files`.
- **`.gitignore` does not support inline comments.** `docs/foo.md  # note` is a literal pattern
  that matches nothing. Put comments on their own lines.
- **Test what you run.** A passing test is not the same as "the probe measures what its
  docstring says". Verify the measurement, not just the arithmetic.
- **Server restarts kill long background jobs.** Use a stable log path outside the harness's
  rolling shell directory; redirect with `> file 2>&1`.
- **Colab truncates the middle of long cell outputs.** The JSON on Drive is the record; the
  printed output is a convenience.
- **`results/` and `data/` are gitignored.** They must be symlinked from Drive in the session
  script, or `ProbeContext` fails before the first gradient.
- **`run_id` collisions.** BP-STE and BP-SSTE share an id; PCKPT and Tier B share an id.
  Do not run both into the same `results/` dir without fixing R9.
- **Never claim a gate is passed until the measurement matches the claim.**
- **No test-set tuning.** Selection by validation loss only.

## The budget

- Burn rate: **1.94 CU/h** (owner-observed 2026-10-31). T1's 1.52 is retained for history.
- Allowance: 300–400 CU/month.
- Every session script logs `--cu-burn-rate 1.94`.

## Build / test / run

```powershell
# compile
.venv\Scripts\python -m py_compile dustq/*.py scripts/*.py tests/*.py

# probe suite (slow: ~1.5 h CPU)
.venv\Scripts\python -m pytest tests/test_probes.py -q --tb=short
# fast subset
.venv\Scripts\python -m pytest tests/test_probes.py -k "not (reference or ladders or cli_main)" -q

# T5 (14 tests, ~1 min)
.venv\Scripts\python -m pytest tests/test_t5.py -q

# phase 0 bitwise parity
.venv\Scripts\python -m pytest tests/test_phase0.py -q

# rebuild the handoff zip (always after a commit)
git archive -o dustq.zip HEAD

# commit (PowerShell quoting trap)
git add -A
# write message to a file, then:
git commit -F C:\Users\shahe\AppData\Local\Temp\opencode\commit_msg.txt
git archive -o dustq.zip HEAD
```

## The Colab probe session

```powershell
# 1. rebuild the zip
git archive -o dustq.zip HEAD
# 2. upload dustq.zip to Colab
# 3. paste scripts/colab_probe.py into ONE cell on an L4 runtime
```

~5–6 CU, ~3 h. Paste back `results/probes/*_probes_*.json`.

## Code layout

- `dustq/config.py` — `RunConfig`, `load_config`, `SIZES`, `CSV_COLUMNS`. `run_id` is
  derived (not a constructor arg). `load_config` pops `run_id` from the YAML.
- `dustq/quant.py` — `ste_act_quantize`, `sste_act_quantize`, `QLinear` (fake-quant weight +
  identity STE; exact integer-code path gated by `_exact_path` + `is_grad_enabled()`),
  `_tf32_int_matmuls`, `set_exact_mode`.
- `dustq/qmodel.py` — `parse_precision`, `swap_quantized_linears`, `register_act_hooks`,
  `site_sigma`, `site_sigma_key`, `site_unsigned`, `calibrated_alpha`, `set_exact_path`.
- `dustq/probes.py` — probe suite 0–5, `ProbeContext`, `cosine_by_group`,
  `probe_cosine_ladders`, `probe_sigma_delta`, `probe_directional_descent`, `main()`.
  Probe 1 returns `(ref, r_g, meta)` where ref is the averaged two-half reference.
- `dustq/train.py` — `Runner`, `_build`, `_load_data`, `evaluate`, `load_best`.
- `dustq/runner.py` — `build_model`, `tiny_batch`, `run_dust`, `run_backprop`.
- `dustq/dust_ext.py` — `DUSTQ(DUST)` with configurable sigmas.
- `dustq/recipes.py` — `recipe()` for L2/L4/L8.
- `dustq/queue.py` — `run_queue` (idempotent by `run_id`).
- `dustq/backprop_ext.py` — `make_bp_optimizer`, `_bp_step`, `train_step`.

## Conventions

- **Claim tags** in docs: `[paper]`, `[code]`, `[design]`, `[unverified]`.
- **Commit messages** via temp file (`git commit -F`).
- **Reports** in `docs/reports/` per task (T0, T1, T3, T4, T5, T6).
- **Findings** appended to `docs/FINDINGS.md` (internal).
- **Open questions** numbered in `docs/OPEN_QUESTIONS.md` (internal).
- **Internal docs** are untracked (see `.gitignore`). Don't commit them.
- **Upstream** is never edited. All behaviour lives in `dustq/`.

## The traps specific to this codebase

1. **The probe reference is not K=16384.** `probe_reference` caps at K=4096 preset (recipe
   only supports {256, 1024, 4096, 16384} and 16384 is too slow). The "K=16384 reference" is
   a K=8192 averaged estimate. State what you divided out.
2. **`site_sigma` returns Dust's defaults, not per-token scales.** The SSTE σ is absolute
   in activation units (0.05–0.4), not code units. On the A8 grid σ/Δ = 5–40, which is the
   regime where the old SSTE backward broke.
3. **The `mse_alpha` scale is `alpha * mean|x|` per token, not absmax.** The floor
   `clamp(min=1e-5)` is now in place; keep it.
4. **`QLinear.forward` takes the exact path only under `no_grad()`.** Under grad, it's
   `F.linear(x, effective_weight())` regardless of `_exact_path`. So the exact path is
   DUST-only by construction.
5. **`cfg.dtype` is honoured only on the dust path.** The backprop path hardcodes bf16
   autocast (R11 — open).
6. **`RunConfig.run_id` is derived.** Don't pass it to the constructor.
7. **`load_config` pops `run_id` from the YAML.** It's informational only.
8. **The `group_of` function must strip `.weight`** and match both suffix and prefix for
   `value_embeds`. Fixed; don't regress.
9. **`probe_cosine_ladders` reliability defaults to 1.0** when `r_g` is not passed. Always
   pass it from the CLI.

## Where the science stands (as of 2026-11-13)

- **G0** (reproduction): passed, one documented exception (K=1024 single-seed spike).
- **G1** (dtype): passed on both criteria (gradient + loss).
- **G2a**: arm 2 (descent) passed with the *wrong* direction (per-tensor norm). Arm 1
  (ladder rise) was lost twice over. **Not readable.**
- **T5**: code complete, alpha table frozen, PCKPT checkpoints on Drive.
- **Watch**: Dust has one bibliography-only arXiv citation (`2610.33826v2`). Noise benchmark
  record 3.49652. Fork frozen. Semantic Scholar blocked (manual check is the hard gate).
- **OQ-40**: undecided. Recommendation `fake` for training, exact paths as artifacts.
- **OQ-41/38**: Semantic Scholar manual check outstanding.

## The single most important sentence

**Nothing from the pre-fix cohort is comparable to anything post-fix.** The reference
convention, the group set, the flip-rate basis, the descent normalisation and the
disattenuation divisor all changed in one week. Every claim about the probes is provisional
until the re-run lands.

See `docs/HANDOFF.md` for the full narrative.
