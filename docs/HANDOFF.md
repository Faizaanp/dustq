# HANDOFF — Dust-Q (as of 2026-11-13)

This is the single document to read if you are picking up this project. It records what has
been done, what is wrong, what is left, and how to operate it. Everything else in the repo
is supporting material.

**Repository state at time of writing:** `main` at `1acb72e` (after the internal-docs cleanup
and the remediation-report untrack), `dustq.zip` rebuilt at 242,391 bytes. Working tree clean.
Public tracked docs: 10 files (see §4). Internal docs are on disk but untracked.

---

## 1. What this project is

**Goal:** extend Dust (Q Labs' backprop-free pretraining via activation perturbation) to
**quantized transformers (ternary/int8 weights, A8/A4/A2 activations) trained from scratch**,
targeting arXiv then TMLR. Plan: `docs/planning/Dust-Q Implementation Plan v3.md` (v3.1,
supersedes v1/v2).

**The scientific question:** does Dust's credit assignment survive quantization? Tested by
comparing Dust's gradient estimate against backprop+STE (BP-STE) and a Gaussian-smoothed STE
control (BP-SSTE), at matched budgets, across precisions P0 (fp) → P1 (w8a8) → P2 (w1.58a8)
→ P3 (w1.58a4) → P4 (w1.58a2).

**Where we are in the plan:** Phase 0 done (reproduction), T5 done (quantization
implementation), G1 closed (dtype gate). **G2a is NOT readable** — see §5. Tier B (the
training matrix) has not run. Nothing has been submitted.

**Owner rules (binding):**
- No model execution on the owner's machine. Local runs are the test suite and smoke steps
  only; training, calibration, benchmarking and probing happen on Colab L4.
- Burn rate: **1.94 CU/h** (owner-observed 2026-10-31), replacing T1's 1.52 baseline.
- Compute allowance: 300–400 CU/month.

---

## 2. The rules of engagement

### v1 agent rules (still in force)
- **Never invent numbers.** Tag every claim: `[paper]` (verified from a primary source),
  `[code]` (measured), `[design]` (a choice), `[unverified]` (not yet checked).
- **No test-set tuning.** Selection is by validation loss only.
- **Log every run, including failures.**
- **Per-task report in `docs/reports/`.**
- **Upstream is vendored unmodified** in `third_party/dust/`, pinned at commit
  `b20f7c03eac630b6441ba6f254128c1761dc45ad` (MIT, `UPSTREAM_COMMIT` file).

### Operational rules learned the hard way
- **Stale zips cause phantom bugs.** Always `git archive -o dustq.zip HEAD` after every
  commit and re-upload. A stale zip produces bugs that look like science.
- **PowerShell quoting trap.** Commit messages must go through a temp file:
  `git commit -F <file>`. Inline quotes corrupt multi-file commits.
- **Swallowed errors.** Any subprocess that matters gets captured and its tails printed on
  failure. An inner crash that vanishes is indistinguishable from success.
- **Write data before pretty-printing it.** Persist first, cosmetics second.
- **No test-set tuning** and **no fabricated claims** — see the 2026-11-06 watch correction
  for why this matters (we printed "Dust still uncited" as fact; arXiv full-text found one
  bibliography-only citation).

---

## 3. Repo layout

### Public (tracked, ships in `dustq.zip`)
- `dustq/` — 12 modules: `config.py`, `train.py`, `runner.py`, `quant.py`, `qmodel.py`,
  `probes.py`, `dust_ext.py`, `backprop_ext.py`, `recipes.py`, `queue.py`, `upstream.py`, `__init__.py`
- `tests/` — 6 files, 59 test functions (57 collected as 65 items incl. parametrized)
- `scripts/` — 9: `colab_probe.py`, `colab_tier_r.py`, `colab_p2.py`, `benchmark.py`,
  `calibrate_alpha.py`, `make_configs.py`, `run_queue.py`, `colab_probe.py`, `emit_*.py` (scratch)
- `configs/` — 26 YAMLs: `alpha_calibrated.json` (frozen), 5 PCKPT configs per precision, etc.
- `third_party/dust/` — 12 files, pinned upstream (MIT)
- `docs/` — 10 public files: `HYPOTHESES.md`, `REPO_MAP.md`, `paper_tables.md`,
  `related_work.md`, plan v3.1, and gate reports `T0/T3/T4/T5/T6` (T1 and the perf
  addendum are internal)
- `README.md`

### Internal (on disk, untracked — do not commit)
`docs/OPEN_QUESTIONS.md`, `FINDINGS.md`, `MONITORING.md`, `CUT_CELLS.md`,
`docs/reports/WATCH/*`, `docs/reports/STATUS-*`, `docs/reports/REVIEW-*`,
`docs/plans/*`, `docs/specs/*`, `docs/reports/T1.md`, `docs/reports/T5-addendum-perf.md`,
`docs/REMEDIATION-2026-11-08.md`, `docs/HANDOFF.md` (this file), `docs/AGENTS.md`,
`data/`, `results/`, `runs/`, `temp/`, `*.zip`, `*.pt`.

### Key files of record
- `docs/reports/T3.md` — G0 verdict (reproduction)
- `docs/reports/T4.md` — G1 verdict (dtype) + loss confirmation
- `docs/reports/T5.md` / `T5-addendum-perf.md` — T5 implementation + perf
- `docs/reports/T6.md` — probe session, six bugs, first P3 readings
- `docs/reports/WATCH/2026-11-06.md` — the re-scan that found Dust's citation
- `docs/FINDINGS.md` — the append-only findings log (internal)
- `docs/REMEDIATION-2026-11-08.md` — the 22-item audit (internal)
- `docs/OPEN_QUESTIONS.md` — 44 items, the decision log

---

## 4. What has been done

### T0–T3 / G0 — reproduction ✅ (with one documented exception)
- Reproduced Dust on M/1M and S/1M. 11/12 seeds within 0.020 of Appendix C. The exception:
  K=1024 single-seed spike-recovery event (delta 0.102, recovered). See `docs/reports/T3.md`.

### T4 / G1 — dtype gate ✅ closed on both criteria
- Gradient cosine +0.9910 (mean, all groups), worst group +0.938 (c_k), lm_head 1.0000.
- Loss rows: P0 fp32 6.0620 (inside the 3-seed bf16 spread 6.0547–6.0680, +0.0017 ≈ 0.25σ);
  P3 fp32 6.1119 vs P3 bf16 6.1126 (delta 0.0007, two orders below seed noise).
- bf16 is usable for the fp path on the L4.

### T5 — quantization implementation ✅ code-complete
- `dustq/quant.py`: fake-quant STE, SSTE (Gaussian-smoothed backward), TF32 integer-code
  matmuls (`_tf32_int_matmuls`), code caching (`_codes`/`_codes_version`), fp32-latent assertion.
- `dustq/qmodel.py`: `QLinear` swap, act-quant pre-hooks (absmax / mse_alpha schemes),
  `site_sigma`, `calibrated_alpha` (frozen table lookup), `set_exact_path(mode=)`.
- `dustq/train.py` / `runner.py`: `Runner` for all methods, resume, atomic `best.pt`.
- Alpha table frozen at `configs/alpha_calibrated.json` (two-stage log-grid search).
- PCKPT checkpoints (P0/P1/P2/P3) trained and on Drive.
- Exact-path perf: TF32 gave only +3–7% over fp32; caching was the real win (2.48–3.18×
  pre-T5). See `docs/reports/T5-addendum-perf.md` (internal).

### T6 — probes ✅ code-complete, ❌ not yet run successfully on the L4
- `dustq/probes.py`: probe suite 0–5, `ProbeContext`, `cosine_by_group`, `fit_cos_law`,
  `probe_directional_descent`, `probe_averaging_test`, `probe_per_tensor_cosine`.
- All four audit bugs fixed: hook ordering, sigma keys, median Δ, flip rate.
- Reference is now the averaged `(g1+g2)/2` (K=8192), divisor `sqrt(2r/(1+r))` per group.
- CLI (`main()`) writes per-probe-set JSONs with save-after-every-probe.
- 15 probe tests in `tests/test_probes.py`.

### Watch / literature
- 6 watch editions (through 2026-11-06). Dust has exactly one arXiv citation
  (`arXiv:2610.33826v2`, bibliography-only). Semantic Scholar blocked five runs.
- Noise benchmark: zero quantized entries; record now 3.49652 (PR #56, 2026-11-05).
- Fork `quant-int8-relay` unchanged at `fdaca52`, 0 forks.
- JacQuant v1 only.

---

## 5. What is wrong right now (read before quoting anything)

The 2026-11-08 audit found four bugs in code I wrote and verified this week, all confirmed
by data, all now fixed. They invalidate specific claims:

1. **Hook order** (fixed, `34b5ad7`): `dust_grads` registered act hooks before `DUSTQ`,
   so Dust's `o@l` jitter was dropped on the exact paths (c_q/c_k/c_v grads → exactly 0.000000)
   and post-quantization on the fake path. **Every pre-fix probe cosine is on a different
   function from what Tier B trains.**

2. **σ–Δ flip rates were a bin-centred artifact** (fixed, `f67533e`): the spy captured the
   quantized input, so every activation sat at its bin centre instead of a continuous offset.
   **T6's "sub-¼Δ moves essentially no A4 codes" is retracted.** True continuous rates are
   ~0.10 at ⅛Δ and ~0.20 at ¼Δ.

3. **Disattenuation divided by r, not √r** (fixed, `bf0c10d`): the reference was one K=4096
   draw, not K=16384. The "consistent with perfect agreement at writer/hub/head" claim is
   overstated; the corrected numbers are lower. The corrected ladder JSONs are in
   `results/probes/corrected/`.

4. **Directional descent normalised per-tensor** (fixed, `f67533e`): each parameter was
   stepped by its own L2 norm instead of the global one, so the tested direction was a
   distorted `d`, not the gradient. **G2a arm 2's deltas are wrong.**

Plus two additional findings whose physics is unverified:
- **q/k/v low agreement is present at fp** (K 0.106 at P0), so it is *not* a quantization
  effect. It's structural to Dust-vs-backprop at this scale.
- **P1/P2 BP-SSTE collapses** (Attn out −0.045, Q 0.012 at P2; Attn out 0.041, K −0.007
  at P1) while BP-STE stays healthy. Root cause found: `sste_act_quantize`'s backward had
  two defects — a `1/Δ` scale error (100× at A8) and a fixed ±3 boundary truncation.
  **All SSTE training results are suspect until re-checked.**

**G2a is not readable.** Nothing from the pre-fix cohort is comparable to post-fix numbers.

---

## 6. What is left (prioritized)

### P0 — blocks the gate
1. **Re-run the full probe session** with the fixed code. `dustq.zip` is at `ac2cbc3`.
   Steps: P3 `3,2,0` → `4` → `5`, then P0/P2/P1 `3,2`, then M `2,5`. ~5–6 CU, ~3 h.
   Paste back `results/probes/*_probes_*.json`. This produces the first internally consistent
   dataset.
2. **R5 reshape** in `calibrate_alpha.py`: per-token `(N, C)` instead of global mean. Re-run
   calibration and re-freeze the alpha table. Cheap, Colab-only.
3. **R6 honest rewrite** of T6's SSTE record: both P1 and P2 collapsed; "directions survive"
   is retracted.

### P1 — Tier B hazards
4. **R8 L4 reproduction** of the bf16-autocast claim on the exact path (needs GPU).
5. **R9 `run_id` collisions** — BP-STE vs BP-SSTE and PCKPT vs Tier B will silently skip
   runs in `run_queue`. Add `ste_mode`/`task`/`sigmas` to the id.
6. **R10 BP-SSTE ignores `cfg.sigmas`** — breaks the matched-σ contract.
7. **R11 `dtype=fp32` ignored on the backprop path.**
8. **R13 `probe_averaging_test` bias formula** (subtract `ref_norm`, not `ref_norm/2`).
9. **M1 `calibrate_alpha` reshape** (same as R5).
10. **R12 clamp** (done) + half-width cap (still open).
11. **R14–R19** — partial-subset resume hazard, benchmark `IndexError`, concurrency
    measurement invalidation, VRAM leak, README staleness, HYPOTHESES early-exit rule.

### P2 — hygiene (before any push)
12. **R20 Kaggle API key in git history** (`fdb9114`, `6be678c`). Rotate it, then decide on
    `git filter-repo` (rewrites all descendant hashes — see the warning in
    `docs/REMEDIATION-2026-11-08.md`). **Owner decision, not done unilaterally.**
13. **R21** untrack `example_task.py` and `kaggle_benchmarks_reference.md` (tracked, shipped
    in the zip, unrelated to Dust-Q).
14. **R22** corrected-ladder schema mismatch (done in code; output files need regeneration).
15. **R13** averaging-test bias formula.

### P3 — owner decisions
16. **OQ-40** (forward mode): recommendation `fake` for all training runs, exact paths as
    correctness artifacts + one ablation. Still undecided.
17. **OQ-41/38** (Semantic Scholar citation check): hard pre-posting gate, manual check needed.
18. **OQ-31** (Noise benchmark hygiene): re-quote before posting (record now 3.49652).
19. **Tier B scoping**: re-decide after G2a lands.
20. **The `group_of` / norm-ratio reporting** I proposed but never implemented — worth doing
    before the session so the ladder JSON carries the magnitude data.

---

## 7. How to run things

### The probe session (the next thing)
```
# 1. rebuild (always, after every commit)
git archive -o dustq.zip HEAD
# 2. upload dustq.zip to Colab
# 3. paste scripts/colab_probe.py into ONE cell on an L4 runtime
```
~5–6 CU, ~3 h. Paste back `results/probes/*.json`.

### The test suite
```powershell
# Full probe suite (slow, ~1.5 h)
.venv\Scripts\python -m pytest tests/test_probes.py -q --tb=short
# Fast subset (the three slow ones are the reference, ladders, cli e2e)
.venv\Scripts\python -m pytest tests/test_probes.py -k "not (reference or ladders or cli_main)" -q
# T5 (14 tests, ~1 min)
.venv\Scripts\python -m pytest tests/test_t5.py -q
# Phase 0 bitwise parity
.venv\Scripts\python -m pytest tests/test_phase0.py -q
```

### Training a checkpoint (Colab only)
```bash
python -m dustq.train --config configs/<run_id>.yaml
```

### Calibrating alpha (Colab only)
```bash
python scripts/calibrate_alpha.py --checkpoint results/<run_id>/best.pt --batches 8
```

### Committing (PowerShell quoting trap)
```powershell
git add -A
# write message to a file, then:
git commit -F C:\Users\shahe\AppData\Local\Temp\opencode\commit_msg.txt
git archive -o dustq.zip HEAD
```

---

## 8. Known traps

1. **Stale zips.** Always rebuild after every commit.
2. **PowerShell quoting.** Inline commit messages get corrupted. Use `git commit -F`.
3. **Swallowed errors.** Any subprocess that matters gets captured; tails printed on failure.
4. **Write data before pretty-printing.** Persist first.
5. **Test what you run.** The probe suite must run off-L4 (it now can, via `data_dir: synthetic`).
6. **No model execution on the owner's machine.** Tests and smoke steps only.
7. **Never claim a gate is passed until the measurement matches the claim.** (This session's
   lesson, stated explicitly.)
8. **Corrections sit next to the original.** The audit trail is the value.
9. **`git rm --cached` alone never sticks.** Add the `.gitignore` rule first.
10. **Server restarts kill long background jobs.** Use a stable log path and redirect
    output to a file outside the harness's rolling shell directory.
11. **Colab truncates the middle of long cell outputs.** The JSON on Drive is the record;
    the printed output is a convenience.
12. **The disattenuated column was silently the raw one.** Now fixed and recorded as
    `disattenuation_applied` (now `disattenuation` with per-group divisors). Old JSONs are
    not comparable.
13. **`results/` and `data/` are gitignored.** They must be symlinked from Drive in the
    session script, or ProbeContext fails before the first gradient.
14. **`run_id` collisions.** BP-STE and BP-SSTE share an id; PCKPT and Tier B share an id.
    Don't run both into the same `results/` dir without fixing R9.
15. **The Kaggle API key** is in git history (`fdb9114`, `6be678c`). Rotate before any push.

---

## 9. Open questions needing owner decisions

- **OQ-40**: forward mode (`fake` recommended).
- **OQ-41/38**: Semantic Scholar citation check (manual).
- **OQ-31**: Noise benchmark re-quote before posting (record now 3.49652).
- **OQ-37/39**: averaging-test probe (implemented; instrumentation decision).
- **R20**: key rotation and history scrub.
- **Tier B scoping**: re-decide after G2a lands.
- **R18**: HYPOTHESES early-exit rule (P4 becomes primary).

---

## 10. Files of record (the scientific trail)

| file | contents |
| --- | --- |
| `docs/reports/T3.md` | G0 verdict: reproduction, 11/12 seeds within 0.020 |
| `docs/reports/T4.md` | G1 verdict: bf16 usable, loss confirmation |
| `docs/reports/T5.md` | T5 implementation, perf regression, TF32 fix |
| `docs/reports/T6.md` | Probe session, six bugs, first P3 readings, retracted claims |
| `docs/reports/WATCH/2026-11-06.md` | Dust citation found; Noise/Fork unchanged |
| `docs/FINDINGS.md` | Append-only findings log (internal) |
| `docs/OPEN_QUESTIONS.md` | 44 open items (internal) |
| `docs/REMEDIATION-2026-11-08.md` | 22-item audit + plan (internal) |
| `docs/HYPOTHESES.md` | Pre-registered endpoints (public) |
| `docs/planning/Dust-Q Implementation Plan v3.md` | Canonical plan (public) |
| `docs/related_work.md` | Positioning + watch deltas (public) |
| `docs/REPO_MAP.md` | Upstream mapping (public) |
| `configs/alpha_calibrated.json` | Frozen alpha table (public) |
| `results/probes/corrected/*.json` | Disattenuated ladders, corrected (on disk) |
| `results/probes/*.json` | The L4 session's probe output (on disk) |

---

## 11. The single most important sentence

**Nothing from the pre-fix cohort is comparable to anything post-fix.** The reference
convention, the group set, the flip-rate basis, the descent normalisation and the
disattenuation divisor all changed in one week. Every claim about the probes is provisional
until the re-run lands, and the re-run must produce its own baseline. Do not mix cohorts
in the paper, in the findings log, or in conversation. If a number's provenance is unclear,
it is unknown, not measured.

---

*Written 2026-11-13. The next entry in this file should be a dated note recording what the
re-run produced.*
