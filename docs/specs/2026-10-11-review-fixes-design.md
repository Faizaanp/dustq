# Review-fixes design (spec) — 2026-10-11

Input: `docs/reports/REVIEW-2026-10-11.md` (independently verified by probe; all blocking and
correctness claims reproduced — C1 measured at 22.9% A8 code divergence, worse than the
review's 10.5%). Design approved by the owner 2026-10-11. **Revised 2026-10-14: the owner's
second review (8 claims) verified against the code and incorporated in full** — benchmark
C5-guard exemption (claim 1), Colab sanity composition (2), GPU tolerance vs bitwise (3),
C1 residual-bf16 caveat (4), `load_best()` helper (5), `--retry-failed` (6), M1/M3 test
gaps (7), C2 budget fallback (8). Nothing below is implemented until
the plan is approved.

Goal: make the first Colab L4 session productive — fix B1–B5 and C1–C3, add the guards and
measurement-validity fixes, and the three doc corrections. Deferred to T5 (unchanged): C4
clip-gradient policy, MSE-calibrated α_b, integer-code matmul, BP-SSTE.

## Batch A — blocking (B1–B5)

| ID | Change | File(s) |
| --- | --- | --- |
| B1 | Add `n_kv_head = n_head` to all four `SIZES` entries | `dustq/config.py` |
| B2 | Alias map `S→L4d256`, `M→L8d512` for `--sizes` (both spellings accepted) | `scripts/benchmark.py` |
| B3 | `parse_precision` normalizes both `w1.58a4` and `w1p58a4` spellings; benchmark default precisions become `fp,w1.58a4` | `dustq/qmodel.py`, `scripts/benchmark.py` |
| B4 | `swap_linear` constructs `QLinear(..., device=module.weight.device, dtype=module.weight.dtype)` | `dustq/quant.py` |
| B5 | Dust path injects `dataclasses.replace(settings, lr=cfg.lr, momentum=cfg.momentum)` before `make_optimizer` | `dustq/train.py`, `dustq/runner.py` |

## Batch B — correctness and measurement validity (C1–C3, C5, M1–M7)

- **C1** `ste_act_quantize`: compute on `x.float()`, return fp32. On CUDA, autocast casts
  per-op for the matmul; DUST's capture already stores `args[0].float()`. The zero-noise
  bitwise test is unaffected (both sides use the same hooks). *Residual caveat (review 2):*
  this fixes the quantizer's rounding decisions, not the matmul's input precision —
  autocast still casts the dequantized values (codes × scale, not exactly representable)
  to bf16 before the matmul, so grid distortion persists until T5's integer-code matmul.
  Interim quantized results are therefore approximate; G1's bf16-vs-fp32 gate covers the
  question, and the G1-fail fallback (fp32 quantized cells, ~2× cost) removes it.
- **C2** `_load_data` (file mode): load `max(cfg.steps, budget_steps) * 8` rows — where
  `budget_steps` is the upstream default for `cfg.tokens` — shuffle once by `seed+1`, then
  run `cfg.steps`. Truncated and full runs now share batch k. Synthetic mode unchanged.
  The `budget_steps` lookup falls back to `cfg.steps` when the token budget has no upstream
  entry (e.g. a future 20M budget), keeping the schedule well-defined (review 2, minor).
- **C3** `best.pt` becomes self-describing and atomic: one file `{"best": float, "state":
  state_dict}` written via `tmp.replace()`; on resume `best = min(ckpt["best"],
  best_file["best"])`; `_finalize` reads both the weights and the best value from the same
  file. *Deviation from the approved sketch:* I had proposed a separate `best_val.json` to
  preserve upstream's bare-state_dict format, but two files cannot be written atomically —
  a crash between the two writes reintroduces exactly the bug being fixed. Single-file is
  strictly simpler and closes every crash window. Format change documented in REPO_MAP.
  Add `dustq/train.py::load_best(path, map_location) -> (best, state_dict)` as the only
  sanctioned reader: `_finalize` uses it, and the T6 probes must too — nothing else may
  `torch.load` best.pt directly (review 2: `_finalize` at train.py:198 currently loads it
  bare and would break).
- **C5** `register_act_hooks(..., allow_lowbit_absmax=False)`: raises on `bits <= 4` with
  absmax unless allowed; `Runner` allows when `cfg.task == "smoke"`; the smoke harness
  (`runner.run_dust/run_backprop`) passes allow=True; `scripts/benchmark.py` also passes
  `allow_lowbit_absmax=True` — its `make_job` hooks the quantized forward directly and
  timing does not depend on the scale scheme, so without the flag the guard would crash the
  benchmark's default `w1.58a4` cells (review 2). A2/A4 real cells stay blocked until
  T5's α_b lands.
- **M1** queue: failed-attempt marker `run_dir/FAILED_ATTEMPTS` (count + last error); a
  config with ≥2 failed attempts is skipped in later sessions (logged as `skip (failed
  2x)`); no row is ever deleted. `scripts/run_queue.py --retry-failed` resets the marker
  for matching runs and re-runs them (review 2: without an override, anything that failed
  before these fixes would stay skipped forever); the marker is also deletable by hand.
- **M2** on resume, truncate `metrics.jsonl` to lines with `step <= start` (rewrite).
- **M3** 2-concurrent measurement via two `subprocess` workers (`benchmark.py --worker ...`
  mode: runs N steps of one job and prints per-step times), not threads sharing one CUDA
  stream.
- **M4** `colab_t1.py`: capture `t0` at cell start for the wall print. Sanity step is three
  parts, each with a distinct job (review 2: the Phase 0 suite builds on CPU —
  `build_model` defaults `device="cpu"` — so on Colab it verifies code integrity, *not* the
  CUDA path): (i) full `test_phase0.py` (integrity, runs CPU even on Colab); (ii) the
  `test_t1.py` resume subset — `Runner` picks CUDA when available, the in-repo path that
  would have caught B4; (iii) a new CUDA-only test doing one tiny DUST step + one BP-STE
  step per precision (fp, w1.58a4) on the GPU — the direct B4-class check.
- **GPU assertions (review 2):** resume/bitwise equality stays exact on CPU (the recorded
  T1 acceptance evidence). On CUDA, weight comparisons use `torch.allclose` tolerances
  instead of `torch.equal` — SDPA backward uses atomics in some kernels and bf16 ops are
  not guaranteed run-to-run deterministic. GPU resume is a smoke under tolerance; CPU
  bitwise remains the acceptance criterion.
- **M5** REPO_MAP L91 relabel: M/1M + M/10M targets are Appendix C; the S/10M value 5.362
  is main-post Figure 4 (size-study protocol).
- **M6** HYPOTHESES L39: n = 5 seeds for the primary cell and its fp counterpart,
  unconditional (D2 is approved; the conditional phrasing allowed optional stopping).
- **M7** T1.md addendum: points at the review and this pass; the "picked up unchanged"
  claim is superseded.

## New tests

1. Forward every `SIZES` entry (incl. L16/d1024) at seq 64 — B1.
2. `benchmark.py --scale tiny` end-to-end as a subprocess (fp + w1.58a4) — B2/B3 class.
3. Swap preserves dtype (CPU) and device (CUDA-if-available) — B4.
4. Dust optimizer base group equals cfg lr/momentum after override — B5.
5. bf16 input to act quantizer → codes equal fp32-math codes; output dtype fp32 — C1.
6. Truncated schedule == full-run schedule prefix (skipif no data files) — C2.
7. Stale-ckpt resume does not overwrite best.pt with worse weights; final best.pt matches
   the uninterrupted run — C3.
8. bits ≤ 4 + absmax raises unless allowed — C5.
9. No duplicate steps in metrics.jsonl after resume — M2.
10. Failure-marker lifecycle (M1, review 2): a deterministic failing config (missing data
    dir) appends failed rows twice, then is skipped as `skip (failed 2x)` with no new row;
    `--retry-failed` resets the marker and re-runs.
11. Worker-mode smoke (M3, review 2): `benchmark.py --worker ...` at tiny scale completes
    with timing output; the 2-concurrent path returns a positive wall time.
12. CUDA-only quantized step (skipif no CUDA): one tiny DUST step + one BP-STE step per
    precision on the GPU — the B4-catcher for the Colab sanity step.

## Acceptance

- `pytest tests` green (Phase 0 + T1 + new tests) on CPU; CUDA-marked tests skip locally and
  run in the Colab sanity step.
- `benchmark.py --scale tiny --precisions fp,w1.58a4` completes end-to-end locally and
  writes `results/benchmark_cpu.md`.
- `dustq.zip` regenerated from HEAD after all commits.

## Out of scope (unchanged, T5): C4 clip-gradient decision (recorded in HYPOTHESES/Method
when α-clipping lands), α_b MSE calibration, integer-code matmul, BP-SSTE autograd.
