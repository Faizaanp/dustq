# Remediation plan — audit of 2026-11-08

Source: four parallel audit subagents over the repo at `bf0c10d`. Every item below marked
**VERIFIED** was re-checked by hand against the code or the downloaded probe JSONs before
being listed; items marked **UNVERIFIED** were reported but not yet reproduced and need a
reproduction step first. Priority reflects what invalidates published claims, not effort.

Severity key: **P0** = a claim in the paper or a gate verdict is wrong right now ·
**P1** = will silently corrupt Tier B / T7 · **P2** = correct-by-luck or cosmetic ·
**P3** = hygiene, must clear before any push.

---

## P0 — claims and gate verdicts that are wrong today

### R1. Disattenuation divides by r instead of √r (`probes.py` `probe_cosine_ladders`) — VERIFIED

`probe_reference` returns `(g1, g2, r)` and every caller discards `g1`, using `g2` alone as
the reference. The ladder's K=4096 point then recomputes `g1` (`seed=0, K=4096`), so that row
*is* `cos(g1, g2)` — confirmed: `reliability` equals the median group's K=4096 raw cosine to
16 decimal places in all four downloaded JSONs.

Consequences: (a) `disattenuated = raw / r` divides out the reference's noise twice; the
spearman–brown correction for a single half is `√r` (or `√(2r/(1+r))` if the reference were the
*mean* of the halves). (b) The reference is one K=4096 draw, not the K=16384 the docstring
claims, so `reliability` (0.29–0.38 at S, 0.080 at M) is that draw's noise floor, and the
r2=0.962 Head row is a K=4096-with-itself coincidence rather than evidence.

**Fix:** carry both halves. `probe_reference` should return the *averaged* `ref = (g1+g2)/2`
(a genuine K=8192 estimate, not a single K=4096 draw) together with `r_g = cos(g1, g2)` per
group. The disattenuated cosine then divides by the reference's **correlation with the true
signal**, which for an average of two parallel measurements is `sqrt(r_ref)` with
`r_ref = 2·r_g/(1+r_g)` (Spearman–Brown):

```
disattenuated = raw / sqrt(max(2*r_g/(1+r_g), 0))
```

Two edge cases that must be handled explicitly rather than by a floor:
- **`r_g <= 0`** — the two halves are uncorrelated or anti-correlated, the disattenuation is
  undefined, and dividing by an epsilon would fabricate a number. Report the raw cosine and
  mark the group `disattenuation: undefined`, or NaN. Never `/max(r, eps)`.
- **per-group, not one global scalar.** A single median reliability applied to every group is
  what clamped high-SNR groups to 1.000 in the first place (see R1c above).

**Status: IMPLEMENTED** (`dustq/probes.py`, `probe_reference` now returns the averaged
`(g1+g2)/2` reference, per-group `r_g`, and meta; `probe_cosine_ladders` divides by
`sqrt(spearman_brown(r_g))` per group and returns `None` for undefined groups). Locked in by
`test_ladders_disattenuate_by_spearman_brown_sqrt` and `test_spearman_brown_edges`.

⚠️ **Do not "fix" this by dividing by `r_g`** — that is the current bug. Dividing by the
*stepped-up* reliability by mistake divides the noise out twice *and* still under-corrects the
signal by `1/sqrt(r_ref)`. The correction and the divisor must be the same object.
**Also:** state in `docs/reports/T6.md` that pre-fix JSONs used a single-half reference and an
`r` divisor, so old numbers are not comparable — same cohort discipline as the hook-order fix.

### R2. `probe_sigma_delta` flip rate is a bin-centred artifact — VERIFIED, IMPLEMENTED

`_ensure_hooks("identity")` registers the act-quant hook *before* the spy, so the spy's
`args[0]` is already `codes * scale`. Measured: `max |x - codes*scale| = 0.0` and
`|x/scale - codes| = 0.0` exactly. So `(x + noise)/scale = codes + noise/scale` puts every
activation at its bin centre, giving an artificial dead zone — `2(1-Φ(0.5/frac))`, i.e.
~0.00006 at ⅛Δ and 0.0455 at ¼Δ, versus the true continuous rates ~0.10 and ~0.20.

This invalidates the T6 headline "sub-¼Δ noise moves essentially no A4 codes" and the
monotone table built on it.

**Fix:** capture the *pre-quantization* input. The act hook already computes it; stash
`args[0].detach()` in `_act_raw` alongside `_act_codes`/`_act_scale`, and have the spy read
that. Re-run probe 5 after the fix — the whole σ–Δ story (including "the mechanism is now
measured") needs re-deriving from continuous inputs.

### R3. Layer overwrite in `flip_rates` — VERIFIED, IMPLEMENTED

`rates[group_of(name + ".weight")] = ...` overwrites on every block, so only the last block
is reported and `hub_shallow` (blocks 0–1 of 4) is silently dropped. With n_layer=4 that
discards half the layers.

**Fix:** accumulate a list per group and average at the end, not assign.

### R4. `probe_directional_descent` normalises per tensor — VERIFIED, IMPLEMENTED

`ev()` divides each parameter by its *own* tensor norm instead of the global `‖d‖`, so
`resid_lambdas`/`x0_lambdas` (4 elements) are perturbed by the same L2 distance as the
`wte`/`lm_head` matrices. The tested direction is a distorted `d`, not `d` itself. G2a arm 2
as measured is weaker than reported.

**Fix:** compute the global norm of the concatenated direction once, and step each tensor by
`-eps * d_p / ‖d‖`. Also pass `grid` through to `directional_descent(..., steps=grid)` (the
parameter is accepted and dropped).

### R5. `calibrate_alpha.py` pools "per-token" scales globally — VERIFIED (clamp FIXED, reshape still OPEN)

Line 47 does `args[0].detach().float().flatten()`, then line 70 computes
`alpha * x.abs().mean(dim=-1)` on a 1-D vector → one global scalar across all sites, layers
and tokens, instead of per-token `mean|x|` over C. Unit-RMS `norm(x)` inputs and unnormalised
`o@l` inputs are averaged together.

**Fix:** keep `(N, C)` per site (`reshape(-1, C)`), so `mean(dim=-1, keepdim=True)` is
genuinely per-token, then pool *per-token* scales. **R5b — also add `.clamp(min=1e-5)`** to
`quant_mse` in `calibrate_alpha.py` (as `dustq/quant.py`'s absmax and weight paths already do
and as `ste_act_quantize`/`sste_act_quantize` now do, see R12): an all-zero token gives
`scale = 0` and `0/0 = NaN`, which silently poisons the grid search. Re-run the calibration
(cheap, Colab-only) and re-freeze `configs/alpha_calibrated.json`; every A4/A2 number
downstream depends on it — and note the floor means the calibration objective itself changes.

### R6. P1 (`w8a8`) SSTE collapse was never reported; "directions survive" is wrong — VERIFIED, DOCS REWRITE STILL OPEN

The P1 JSON's `sste` column is collapsed (`Attn out 0.041`, `MLP in 0.070`, `resid -0.698`,
`x0 -0.772`) exactly like P2's. T6/FINDINGS report only P2. The reassuring claim "probe
cosines are directions so they survive the scale error" is false: `Δ` is per-token, so
multiplying by `1/Δ_b,t` instead of `Δ_b,t` reweights tokens by `1/Δ_b,t²` in the outer
product and rotates the gradient.

**Fix:** record both precisions as collapsed-pre-fix; retract the "directions survive" wording
in FINDINGS; re-probe P1 and P2 after R8 (already scheduled) and treat every pre-`7dddfb9`
SSTE column as void.

### R7. `test_probes_cli_main_end_to_end` asserts the pre-fix filename — VERIFIED, FIXED

The test checks `<run_id>_probes.json`; `probes.main` now writes `<run_id>_probes_<tag>.json`.
Will show as a false failure once the suite completes.

**Fix:** update the assertion to the tagged name.

---

## P1 — will silently corrupt Tier B / T7

### R8. `QLinear._exact_forward` runs inside active bf16 autocast — HIGH, UNVERIFIED on GPU (R8b clobber FIXED)

Reported mechanism: `DUST.losses`, `evaluate` and `quantized_losses` all run inside
`torch.autocast("cuda", bfloat16)`, and `aten::matmul` is on the bf16 cast list, so
`a @ b` with fp32 integer operands is silently downcast — meaning neither `tf32` nor `fp32`
mode is actually exact on CUDA. The claim also explains T5's perf mystery (no TF32 speedup,
1.4× slower than fake, +12 % VRAM) via the fp32 output promoting downstream ops.

**Reproduce first** (needs the L4): one forward under `--dust-forward tf32` with and without
`autocast(enabled=False)`, compare outputs bit-for-bit. If confirmed, wrap the exact matmul in
`with torch.autocast(device_type=..., enabled=False):` and add a CPU-side unit test for the
same shape of bug.

**R8b — global-state clobber in `_tf32_int_matmuls` — VERIFIED, FIXED.** The helper set
`allow_tf32 = False` unconditionally in `finally`. PyTorch enables `allow_tf32` by default on
Ampere/Ada/Hopper, so this permanently disabled TF32 for every subsequent op in the process —
a silent, global, compounding slowdown, on the exact path we were benchmarking. Now saves and
restores both `allow_tf32` and `fp32_precision` (torch ≥ 2.9 renamed the API); locked in by
`test_tf32_int_matmul_restores_previous_flag`.

### R9. `run_id` ignores `ste_mode`, `task`, `sigmas`, lr/momentum — VERIFIED

`{size}_{tokens}_{precision}_{method}_K{k}_s{seed}[_fp32]` means BP-STE and BP-SSTE collide,
Tier-B seed=0 collides with the existing PCKPT/G1 seed=0 results on Drive (so `run_queue`
will skip them), and the T7 tuning grid collapses onto one id. `CSV_COLUMNS` has no
`ste_mode` or `dust_forward` column either.

**Fix:** add `_ste` / `_dust{fwd}` / task and the tuning discriminator to `run_id`, and the two
missing CSV columns. Migration: do not re-use existing Drive run_ids for new cells; write new
ones and note the scheme change in OPEN_QUESTIONS.

### R10. BP-SSTE ignores `cfg.sigmas` — VERIFIED (by inspection)

`register_act_hooks` reads `DEFAULT_SIGMAS` via `site_sigma`, so T7's per-site σ reaches
`DUSTQ` but not the SSTE control — breaking the matched-σ contract silently.

**Fix:** thread `cfg.sigmas` into `register_act_hooks` and apply per site, mirroring `DUSTQ`.

### R11. `backprop` path ignores `cfg.dtype == "fp32"`; train/eval asymmetry — VERIFIED (by inspection)

`_force_fp32` is only called for `method == "dust"`, so a `dtype: fp32` backprop run still
casts embeddings to bf16 and runs `_bp_step` under bf16 autocast while logging `dtype=fp32`.
And because `evaluate()` is `no_grad`, Dust models take the exact path at eval while backprop
models take the fake path unless `dust_forward="fake"`.

**Fix:** honour `cfg.dtype` on both methods; make the eval forward path explicit rather than
inherited from `_exact_path`.

### R12. `mse_alpha` scale has no floor; SSTE half-width unbounded — VERIFIED, FLOOR FIXED (half-width cap still OPEN)

`mse_alpha` lacks the `clamp(min=1e-5)` that `absmax` has, so an all-zero ReLU² token gives
`0/0`. And `half = ceil(4·σ_u)` uses the batch *max* σ_u, so one tiny-scale token can set a
loop of millions of iterations.

**Fix:** clamp the scale to `1e-5`; bound `half` by the number of interior boundaries
(`hi - lo`), which is the natural cap anyway.

### R13. `probe_averaging_test` bias formula and seed collision — UNVERIFIED

`... - ref_norm/2` where the unbiased value is `ref_norm` (reports a spurious +½‖ref‖
"bias"), and the `t=0` trial reuses the reference's seed so that trial *is* the reference.

**Fix:** subtract `ref_norm`; offset trial seeds so none collides with the reference seed.

---

## P2 — will bite later, cheap to fix

- **R14** `S2` partial-subset resume hazard: `save()` writes after probe 0 and probe 1, so a
  disconnect during probe 2 leaves a file that `colab_probe.py` marks DONE. Fix: validate
  that every expected section for `--probes` is present before counting a step as done, and
  write via `.tmp` + `.replace()`.
- **R15** `S3` `benchmark.py --scale real` IndexError (`r['size']` is never `'S'` after alias
  resolution) and `S4` `colab_tier_r.py` omitting `--worker`, which invalidated the 0.35×
  concurrency figure quoted in OPEN_QUESTIONS/CUT_CELLS. Both are one-line fixes plus a note
  retracting the 0.35× claim until re-measured with `--worker`.
- **R16** `C5` double activation quantization on every forward (the hook computes scale and
  codes, then `ste_act_quantize` recomputes them) — a pure perf fix, worth doing before Tier B.
- **R17** `C6` reference-cycle leak: `gc.collect()` runs with `runner` still bound, and
  `estimator.close()` is skipped on the exception/resume paths. Fix: `del runner` before
  `gc.collect()`, `torch.cuda.empty_cache()`, and `close()` in a `finally`.
- **R18** `D3` the HYPOTHESES early-exit rule ("A4 shows no STE bias → P4 becomes primary")
  has fired on the P0≈P3 STE agreement but no `w1.58a2` config or G2b probe exists. This is a
  scope decision for the owner, not a code fix — flag before Tier B commits.
- **R19** `H3` README is stale: "27 passed, 4 skipped" (actual: 57 test functions / 65
  collected), `~6.8M`/`~37.5M` params (actual 7.34M / 37.75M), and the file tree omits
  `dustq/probes.py`, the colab scripts and `tests/test_probes.py`.

## P3 — must clear before any push

- **R20** `H1` a Kaggle proxy API key is in git history (`fdb9114`, `6be678c`) even though
  `opencode.json` is now untracked. Rotate the key, then scrub with `git filter-repo`. Do this
  before the repo is pushed anywhere.
- **R21** `H2` `example_task.py` and `kaggle_benchmarks_reference.md` are tracked and ship
  inside `dustq.zip`. Untrack and ignore them.
- **R22** `D4` the offline-corrected ladders in `results/probes/corrected/` use a different
  schema and silently drop `ve_gate`/`value_embeds`. **Root cause — VERIFIED, FIXED:**
  `SITE_GROUPS` defined only the eight canonical sites, and `group_of` is a *suffix* match, so
  `transformer.h.{i}.attn.ve_gate.weight` and `value_embeds.{i}.weight` (a *prefix*, a
  top-level module indexed by layer) fell through to their raw parameter names. Anything
  filtering on canonical groups then dropped them, and the group count itself changed — which
  moved the median `reliability` and therefore every disattenuated number. Both are now mapped
  (`"VE gate"` / `"Value emb"`, with an explicit prefix rule for `value_embeds`) and locked in
  by `test_group_of_covers_value_embedding_sites`.

  ⚠️ **This changes the reported group count**, so `reliability` from the old JSONs is not
  comparable to post-fix runs — same cohort discipline as R1. Make the corrected-file schema
  match `*_probes.json` and keep every group.

---

## Explicit checkpoint re-probe matrix (after R1–R4 land)

Which checkpoint needs which probe, so the L4 session does not re-derive numbers already on
Drive. Costs assume ~15 min per probe on the L4 at 1.94 CU/h.

| checkpoint | probes | why | ~CU |
| --- | --- | --- | --- |
| **P3** `L4d256_1M_w1p58a4_backprop_K0_s0` | **2, 3, 4, 5** | both G2a arms, post-fix SSTE, continuous σ–Δ | ~1.5 |
| **P2** `L4d256_1M_w1p58a8_backprop_K0_s0` | **2, 3** | post-fix SSTE; ladder on the new reference | ~0.7 |
| **P1** `L4d256_1M_w8a8_backprop_K0_s0` | **2, 3** | post-fix SSTE (its collapse was never reported) | ~0.7 |
| **P0** `L4d256_1M_fp_backprop_K0_s0` | **2** | ladder on the two-half averaged reference (its SSTE is the identity — nothing to redo) | ~0.35 |
| **M** `L8d512_1M_w1p58a4_dust_K256_s0` | **2** | ladder re-evaluation (its σ–Δ sweep never completed) | ~0.35 |

Total ≈ **3.6 CU**, one session. Nothing else needs the GPU until R5's recalibration, which
is a separate short session on the P0 checkpoint.

## Git-history scrub warning (R20)

`git filter-repo` rewrites `fdb9114`, `6be678c` and **every descendant commit hash**. The
following files cite commit hashes and will become dangling after a force-push:
`docs/FINDINGS.md` (cites `bf0c10d`, `7dddfb9`, `8444306`, `44ab759`, `5b4e795`),
`docs/REMEDIATION-2026-11-08.md`, and `docs/reports/T6.md`.

**Before scrubbing:** record the old→new hash mapping (e.g. in `docs/FINDINGS.md` as a dated
note), and decide whether the repo has a remote to force-push at all. If the owner's copy is
the only copy, the cheaper option may be to rotate the key and leave history alone — the key
is only exposed to anyone with repo access, and the repo is not yet public. **Owner decision,
do not scrub unilaterally.**

## Suggested order

1. **R7** (one line, unblocks the suite) then re-run the full suite undisturbed.
2. **R1–R4** — the four probe-maths bugs. These are the ones that make the current gate
   readings wrong; fix together, then **re-run probes 2, 4 and 5** on the L4. Until then G2a
   is *not* readable and T6's σ–Δ headline is retracted.
3. **R5** — recalibrate α (cheap, Colab-only) and re-freeze the table.
4. **R6** — rewrite the SSTE records honestly: both precisions collapsed, "directions
   survive" retracted.
5. **R8–R12** — the GPU/CUDA and Tier-B hazards, each with a reproduction step first.
6. **R13–R19** — correctness-by-luck and stale docs.
7. **R20–R22** — hygiene, **before** any push or public handoff.

Size: R7, R21 are minutes. R1–R4, R12 are a few hours plus a re-run. R5 is one Colab session.
R8 needs the L4 to reproduce. R20 is irreversible history surgery — last, and only with the
owner's say-so.

## Implementation status (2026-11-08, end of session)

## Implementation status (2026-11-08, end of session)

**Implemented and unit-tested:**
- **R1** — `probe_reference` returns the averaged `(g1+g2)/2` reference plus **per-group**
  `r_g` and meta; `probe_cosine_ladders` divides by `sqrt(spearman_brown(r_g))` and returns
  `None` for undefined groups (`test_ladders_disattenuate_by_spearman_brown_sqrt`,
  `test_spearman_brown_edges`).
- **R2** — the act hook now stashes `module._act_raw` (the pre-quantization input), which the
  spy reads instead of `args[0]`. The flip rate is no longer a bin-centred artifact. **The
  reported flip-rate numbers change**; T6's "sub-¼Δ moves essentially no A4 codes" is
  retracted until re-measured.
- **R3** — `flip_rates` accumulates per group across blocks and averages; no more last-block
  overwrite, so `hub_shallow` is no longer dropped.
- **R4** — directional descent steps by the **global** direction norm (`-eps * d_p / ‖d‖`) and
  passes `grid` through to the line search. G2a arm 2 must be re-measured.
- **R7** — the CLI e2e test asserts the correct tagged filename (`_probes_0-2-3-5.json`).
- **R5b / R12** — `.clamp(min=1e-5)` on the `mse_alpha` scale in `ste_act_quantize`,
  `sste_act_quantize` and `calibrate_alpha.quant_mse`.
- **R8b** — `_tf32_int_matmuls` saves and restores `allow_tf32`/`fp32_precision` instead of
  resetting them to `False`.
- **R22** — `ve_gate` / `value_embeds` are mapped to named groups (prefix rule for
  `value_embeds.{i}`).

**Still open:** R5's per-site reshape, R12's half-width cap, R6's honest rewrite of the SSTE
record, R8's autocast reproduction on the L4, R9–R13, R14–R22.

**Re-probe required before any number above is quoted:** the whole matrix below, because R1,
R2, R3, R4 and the group-count change all alter the reported values.
