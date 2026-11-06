# FINDINGS — the one place to look

**What this file is.** A single append-only log of everything this project has *found out* —
bugs, gate results, watch results, claim corrections, and the process lessons behind them.
Each entry is dated, tagged, and points at its evidence. Nothing here is a plan or a decision;
those live in `docs/planning/`, `docs/OPEN_QUESTIONS.md`, and `docs/CUT_CELLS.md`.

**Why it exists.** Findings were previously scattered across per-task reports, the related-work
file, and a 41-item open-questions list. When a question comes up later ("what happened with
the sigma sweep?", "was Dust ever cited?"), this file should answer it in one place.

**Rules (same discipline as the rest of the repo):**
- Append-only. New entries go at the top of the entries section; never rewrite history. A
  finding later shown wrong gets a dated correction *next to* it, not in place of it.
- Every entry carries a date and one or more tags: `[bug]` `[gate]` `[watch]` `[design]`
  `[process]` `[claim]`.
- No invented numbers or identifiers. If a number was measured, say what it was measured on.
- Every entry links its evidence — a report, a test, a commit, or a fetched URL.

**Tags:** `[bug]` a defect found and (where noted) fixed · `[gate]` a gate reading ·
`[watch]` a literature/ladders/benchmark check · `[design]` a design choice and its rationale ·
`[process]` a way of working, learned the hard way · `[claim]` a claim corrected or clarified

---

## Index

| Date | Tag | Finding |
| --- | --- | --- |
| 2026-11-06 | `[watch]` `[claim]` | Dust has exactly one arXiv citation — `arXiv:2610.33826v2`, bibliography only, no body mentions, no quantization content. The flat claim "Dust still uncited" was false and is corrected. Pre-posting gate still open (S2 429 ×5, Scholar JS-walled). |
| 2026-11-06 | `[watch]` | Noise benchmark still has zero quantized entries; Track-2 record is now **3.49652** (PR #56). OQ-31's staleness warning has bitten — claims quoting 3.991/3.896/3.658 are stale. |
| 2026-11-06 | `[watch]` | The arXiv classic full-text endpoint works where the Atom API's `full_text` mode does not (HTTP 500 since 2026-10-31). A working citation search that had been unavailable for six days. |
| 2026-11-06 | `[watch]` | Fork `quant-int8-relay` byte-identical at `fdaca52`, 0 forks, nothing superseded it. JacQuant still v1, so the charter quotation stands. Six adjacent papers logged; none in our cell. |
| 2026-11-02 | `[bug]` | `results/` was never linked from Drive — gitignored, so absent from the shipped zip — and `ProbeContext` opens `results/<run_id>/best.pt`. Every probe run died before its first gradient, error swallowed by `subprocess.run`. |
| 2026-11-02 | `[bug]` | `dust_grads` registered the act-quant pre-hooks *before* `DUSTQ`, the only Dust path in the repo to do so. On the exact paths this dropped Dust's `o@l` jitter entirely: c_q/c_k/c_v grad norms go to **exactly 0.000000**. On the fake path the jitter propagates but post-quantization (Q/K/V cosines between orders +0.90). |
| 2026-11-02 | `[bug]` | `probe_sigma_delta` built its sigma dict and never passed it — the whole σ–Δ sweep ran at default noise. The dict was also keyed by the probes' own site names, which `DUSTQ` rejects with `ValueError: unknown sigma keys` if wired through as written. |
| 2026-11-02 | `[bug]` | `median_step` hardcoded `/255` (an 8-bit absmax artifact) on A4/A2 `mse_alpha` grids where the code spacing *is* the per-token scale; the promised flip rate was never computed. |
| 2026-11-02 | `[bug]` | A `KeyError` in the σ–Δ table (`sweep[(frac, k)]` with `k` already a `(frac, K)` key) crashed the P3 session **after** its full 16-point sweep had computed. No test exercised `main()`. |
| 2026-11-02 | `[gate]` | **G1 closed on both criteria**: P0 fp32 test 6.0620 (inside the 3-seed bf16 K256 spread, +0.0017 ≈ 0.25σ); P3 fp32 6.1119 vs P3 bf16 6.1126 (delta 0.0007); gradient cosine +0.9910 mean, +0.938 worst group. |
| 2026-11-02 | `[gate]` `[claim]` | First real P3 probe data (provisional): Head 0.94 / Attn out ≥0.77 / MLP out 0.63 / MLP in 0.40 vs K **0.012**, V 0.114, Q 0.237, value_embeds 0.01 — the attention internals are near-orthogonal to the reference while writer/hub/head agree. SSTE is *worse* than STE almost everywhere, so H2's "STE bias is the mechanism" is unsupported. |
| 2026-11-02 | `[process]` | `ProbeContext` demanded `data/train_1M.pt` unconditionally, so the probe suite could not be run off the L4 at all. Every probe bug found to date was invisible locally by construction. Fixed by mirroring `Runner._load_data`'s synthetic fallback. |
| 2026-11-02 | `[process]` | `save()` now runs after every probe and the JSON is written before any pretty-printing, so a reporting bug or session timeout cannot take completed probes' results with it. The session script also runs the G2a gate arms before the expensive sweep. |
| 2026-11-02 | `[process]` | A placeholder identifier for an unsearched watch update was drafted into `related_work.md` and caught before commit. It was removed, and the removal is recorded in the file. |
| 2026-10-24 → 2026-11-02 | `[process]` | Third swallowed-error incident: `colab_probe.py` called `subprocess.run(check=True)` with no capture, so every inner crash message vanished. Now `run_checked()` prints stdout/stderr tails before exiting, plus a preflight block that checks data, the alpha table and all five checkpoints before any GPU time starts. |
| earlier | `[process]` | **Stale zips on Colab cause phantom bugs.** Always re-upload a freshly built `dustq.zip`; rebuild it after every commit (it is a `git archive` of HEAD, so it matches HEAD exactly). |
| earlier | `[process]` | **No model execution on the owner's machine** (2026-10-24 owner rule): local runs are the test suite and smoke steps only; training, calibration and benchmarking happen on Colab. |
| earlier | `[gate]` | **G0 passed** with one documented exception (K=1024 single-seed spike-recovery event, recovered to delta 0.102); 11/12 seeds within 0.020 of published values. See `docs/reports/T3.md`. |
| earlier | `[gate]` | **G1 gradient half passed**: fp32-vs-bf16 Dust gradients on M/1M/K256, mean cosine **+0.9910**, worst group +0.938 (c_k), lm_head 1.0000. See `docs/reports/T4.md`. |
| earlier | `[bug]` | `group_of` didn't strip the trailing `.weight` from parameter names, so every probe reported one group per parameter instead of six site groups. |
| earlier | `[bug]` | `Path(output_dir) / run_id / "best.pt"` raised `TypeError` because `cfg.output_dir` is a string — it would have crashed every probe run before the first gradient. Caught by a CPU test, not by an L4 session. |
| earlier | `[bug]` | The alpha-b `KeyError` (top-level vs nested lookup) failed every A4/A2 run until fixed; the degenerate alpha grid (interior optima pinned to the boundary) fixed by a two-stage search. Both unit-tested. |
| earlier | `[bug]` | The exact integer-code path measured ~1.5× *slower* than the pre-T5 fake-quant path; TF32 gave only +3–7%. Fixed with `_tf32_int_matmuls` (small-integer codes are exact in TF32 inputs, products exact in fp32 accumulation) plus `--exact-mode`. See `docs/reports/T5-addendum-perf.md`. |
| earlier | `[design]` | OQ-40 (Dust vs BP-STE forward consistency): recommendation unchanged — `fake` for all training runs, exact paths as correctness artifacts plus one ablation. JacQuant's charter ("share identical … forward quantization, differ only in backward gradient rules") is the citable precedent. Still the owner's decision. |
| earlier | `[design]` | Burn rate adopted at the owner-observed **1.94 CU/h** (2026-10-31), replacing T1's 1.52 baseline. All session scripts log it; CUT_CELLS and the plan's core line repriced. |

---

## Entries

### 2026-11-06 — watch re-scan, and one corrected claim `[watch]` `[claim]`

Full report: `docs/reports/WATCH/2026-11-06.md` (written by a background watch subagent;
integrated into `docs/related_work.md` in the same commit).

**The material finding is a claim correction, not a novelty threat.** The arXiv classic
full-text endpoint (`arxiv.org/search_classic?searchtype=ft`) — which the Atom API's
`full_text` mode cannot reach (HTTP 500 since 2026-10-31) and which nobody had tried —
returned **exactly one arXiv paper citing Dust**: `arXiv:2610.33826v2` "Reactive Optimizers"
(Bai, Liang, Teoh, Sitzmann; v2 2026-11-04). Full LaTeX HTML read confirms Dust appears
**only in the reference list** (3 occurrences, zero body mentions) and a keyword scan for
`quantiz|int8|ternary|low-bit|1.58` over the whole paper returns no hits.

**Consequences recorded:**
- `docs/reports/STATUS-2026-11-01.md`'s flat "Dust still uncited" was **false as written** and
  is corrected in place with a dated note (original line preserved).
- `docs/OPEN_QUESTIONS.md` #41 now carries the defensible wording: "no citing paper in our
  cell, one bibliography-only citation in a non-overlapping arXiv paper."
- **The pre-posting gate is not closed.** Semantic Scholar failed a fifth consecutive run
  (HTTP 429) and Google Scholar is now behind a JS challenge — and those are precisely the
  channels that catch a citation keyed to "Dahal et al., 2026" without the URL or exact title.
  A manual human-session search remains a hard pre-posting gate.

**Method finding worth keeping:** the arXiv Atom API's full-text mode has been returning
HTTP 500 since 2026-10-31, so every full-text negative from that API for six days was a
*false negative from a broken tool*, not a real absence. The classic UI endpoint works. Any
future "no citing paper" claim must state which endpoint was used.

**Also confirmed unchanged:** `quant-int8-relay` frozen at `fdaca52` (0 forks, a search for the
branch name returns 0 repositories); JacQuant still v1 so the verbatim charter quotation
stands; Noise rules unchanged since the 2026-10-22 rewrite; zero quantization entries across
all 53 Noise PRs — the only low-precision content is *operand* precision inside credit
kernels, and PR #56 records **fp8 operands tried and rejected** ("best case is ~0.05 ms per
estimate").

**Six adjacent papers logged, none in our cell:** MobileSpec `2611.04797` (weight-space ZO ×
INT4 fine-tuning), ZO-Curv `2611.02688` (ZO-as-curvature for a first-order update — its own
framing "ZO gradients break down in pretraining" is citable for our motivation), Greedy Local
Learning `2610.04867v2` (token-budget-matched local-learning pretraining, NeurIPS 2026
workshop), q0 `2606.03938v2` (Dust authors' own 1.8B hyper-epoch pretraining — note the
terminology trap: q0's "population of models" is *not* Dust's "virtual population" of
perturbed activations), plus `2610.38331` and `2610.40051`. One arXiv paper also cites the
Noise benchmark by title (`2611.04488`, Perseus — FHE inference).

**OQ-31's staleness warning has bitten.** The Noise Track-2 record is now **3.49652**
(PR #56, 16 runs, t-test p = 4e-11, median-not-best logged), far past the 3.991/3.896/3.658
figures our claims quoted. Any "we are X below Noise" statement must be re-quoted from the
live README before posting; recorded in OPEN_QUESTIONS #31.

### 2026-11-02 — the probe session, and six bugs found before it ever ran `[bug]` `[gate]` `[process]`

Full writeup: `docs/reports/T6.md`. The Colab session printed its G1 loss rows, then exited 1
at its first probe command with nothing inner printed. Six causes, all fixed, all tested.

**1. The crash: `results/` was never linked from Drive** `[bug]`. `results/` is gitignored, so
`git archive -o dustq.zip HEAD` ships a repo with no results tree, and `scripts/colab_probe.py`
symlinked only `data` from Drive. `ProbeContext.__post_init__` opens
`Path(cfg.output_dir) / cfg.run_id / "best.pt"` → `FileNotFoundError` → exit 1 **before the
first gradient of the session**. Fixed by linking `results` too. The preflight block now
asserts the data files, the alpha table and all five checkpoints before any GPU time starts.

**2. Hook order — the deep one** `[bug]`. `dust_grads` called `_ensure_hooks("identity")`
*before* constructing `DUSTQ`; every other Dust path (`train.py:97→104`, `runner.py:61→66`,
`benchmark.py:37→43`) builds the estimator first, giving noise → quantize. PyTorch forward
pre-hooks chain (verified empirically: each hook sees the previous hook's returned value), so
the probes measured `quantize(x) + noise` instead of `quantize(x + noise)`. Measured with a
hook-order A/B on the exact path:

| site | correct order | reversed order | ratio |
| --- | --- | --- | --- |
| attn.c_q | 0.001697 | **0.000000** | 0.000 |
| attn.c_k | 0.002282 | **0.000000** | 0.000 |
| attn.c_v | 0.002122 | **0.000000** | 0.000 |
| attn.c_proj | 0.001895 | 0.001895 | 1.000 |
| mlp.c_fc | 0.003586 | 0.003586 | 1.000 |
| mlp.c_proj | 0.003158 | 0.003158 | 1.000 |

On the exact paths Dust's `o@l` jitter never reaches the matmul (QLinear computes from codes
taken from the clean input), so the hub-site rewards vanish and `local_errors` has no targets —
the attention-internal credit collapses to exactly zero. On the fake path the jitter still
propagates, but *after* quantization: Q/K/V cosines between the two orders are +0.90, i.e. a
different function rather than a degenerate one. Fixed by constructing `DUSTQ` first and
force-re-registering the act hooks on every call (`DUSTQ.close()` removes Dust's hooks but
leaves ours in place, so a cached registration silently rotates back). Locked in by
`test_dust_grads_hook_order` across two consecutive calls.

**3. The σ–Δ sweep never swept** `[bug]`. The old code built
`sigmas[site] = frac * max(delta, 1e-5)` and then called `ctx.dust_grads(seed=0, population=k)`
with no `sigmas=` argument. The dict was keyed by the probes' own site names (`attn.c_q`,
`mlp.c_fc`, …), which `DUSTQ.__init__` rejects with `ValueError: unknown sigma keys` — so
wiring it through as written would have crashed too. Fixed by adding
`qmodel.site_sigma_key()` as the single source of truth for the site→sigma-key map (the
inverse of `site_sigma`, with the hub split by depth) and passing `sigmas=` per (frac, K).

**4. `median_step` hardcoded `/255`, and the flip rate was never computed** `[bug]`. The A4/A2
grids use `mse_alpha` per-token scales, so the code spacing *is* the scale; replaced with
`median_delta()`. The flip rate the docstring always promised is now computed. Verified on a
tiny smoke checkpoint (mean over the six act sites): monotone in σ —
0.000 / 0.044 / 0.283 / **0.560** / 0.740 / 0.839 / 0.887 / 0.912 at σ = ⅛…16 × Δ — and 0.560
at exactly 1×Δ, the analytically expected magnitude for noise one code-step wide. Sub-¼Δ noise
moves essentially no A4 codes: that is the dead-zone reading.

**5. The reporting bug that ate the sweep** `[bug]`. `sweep[(frac, k)]` where `k` is already
the full `(frac, K)` key → `KeyError: (0.125, (0.125, 256))` — after all 16 points had
computed. No test exercised `main()` (the fixture tests call the probes directly). Fixed to
`sweep[k]`, and more importantly the JSON is written *before* any pretty-printing with
`save()` after every probe, so a reporting bug or session timeout cannot take completed
probes' results with it. `test_probes_cli_main_end_to_end` now covers `main()`.

**6. The probes could not be run locally at all** `[process]`. `ProbeContext` demanded
`data/train_1M.pt` and `val.pt` unconditionally, unlike `Runner._load_data`, which falls back
to `tiny_batch` for `data_dir: synthetic`. Every probe bug above was therefore invisible off
the L4 *by construction*. `ProbeContext` now mirrors the runner's fallback at the runner's
batch/val seeds, so the whole suite runs on CPU in ~30 min. The real sessions are unaffected
(their configs use `data_dir: data`). The PCKPT/`mse_alpha`/`allow_lowbit_absmax=False` path
— which the smoke-only suite never exercised — is covered by `test_pckpt_task_probe_path`.

**G1 closed on both criteria** `[gate]`. From the session's loss rows: P0 fp32 test **6.0620**,
inside the 3-seed bf16 K256 spread (+0.0017 ≈ 0.25σ of the 6.0604 ± 0.0068 base); P3 fp32
**6.1119** vs P3 bf16 **6.1126** (delta 0.0007, two orders below seed noise). Bookkeeping
note: `runs.csv` carries a `failed` row *and* a `completed` row for both w1p58a4 G1 run_ids —
the failed rows are the aborted attempts, superseded by the resume. Full table in
`docs/reports/T4.md`.

**First real P3 probe data (provisional)** `[gate]` `[claim]`. Measured on L4d256 P3, one
batch, `--dust-forward fake`, post-hook-order-fix: writer/hub/head sites agree strongly
(Head 0.937, Attn out ≥0.774, MLP out 0.627, MLP in 0.404, Tok emb 0.505) while every
attention-internal site is near-orthogonal to the K=16k reference (K **0.012**, V 0.114,
Q 0.237, value_embeds 0.010/0.006, ve_gate 0.166/0.295, resid_lambdas 0.048). Probe 3 shows
**SSTE is worse than STE almost everywhere** (Attn out 0.843→0.640, MLP out 0.756→0.650,
Q 0.458→0.406; only x0_lambdas improves, 0.184→0.247) — so H2's "quantization bias in the STE
derivative is the mechanism" is *not* supported. The surviving hypothesis for the
attention-internal collapse is estimator-vs-quantizer interaction at the hub sites (Dust's
σ ≈ 2–4× the A4 code step, so a draw jumps across many codes), which is exactly what probe 5's
σ–Δ sweep settles — and it prints σ per key at 1×Δ so the ratios can be read directly.
**This is a hypothesis about the real checkpoint, not a measurement of it**: the checkpoint Δ's
above came from a tiny fixture, not the L4.

**G2a's actual criterion** (from Plan v3.1 line 310, worth recording because it was misread
once): at P3, cos(Dust-1k, ref) *rises* with K on MLP and writer layers **and** directional
descent along Dust-1k is negative. It is a P3-internal gate — not a fp-vs-quantized comparison.
The cross-precision probes are for the paper's scientific claims, not the gate. The session
now runs the ladder and descent before the sweep, each in its own subprocess, because the
2026-11-02 session crashed inside the sweep and took the descent arm with it.

**G1 cosines are not comparable across the hook-order fix** `[claim]`. G1's +0.9910 is a
bf16-vs-fp32 comparison on one unchanged function, so the dtype verdict stands; the absolute
Dust-vs-backprop probe cosines from before 2026-11-02 are not comparable to post-fix numbers.
Filed as OQ-43 so the cohort is pinned rather than silently mixing two functions.

### Earlier — process lessons, kept because they recur `[process]`

- **Swallowed errors.** Three separate incidents (the truncated-note bug, the probe session's
  exit-1, this reporting crash) all came from inner-process failures whose messages were
  discarded. Rule: any subprocess that matters gets captured, and its tails printed on failure.
  Rule: write data before pretty-printing it.
- **Stale zips.** `dustq.zip` is a `git archive` of HEAD, so it is only as fresh as the last
  build. Rebuild after every commit; a stale zip produces phantom bugs that look like science.
- **Test what you run.** A CPU fixture test caught the `Path(output_dir)` `TypeError` that
  would have killed every probe run. The lesson generalizes: the probe suite now runs off-L4,
  so the next bug in this area gets caught locally too.
- **Flags, not rewrites.** When a finding is corrected later (the "Dust still uncited" claim),
  the original line stays and a dated correction sits next to it. The audit trail is the point.
