# Dust-Q Implementation Plan v3.1 (consolidated)

Prepared 2026-10-10. **Revision 3.1** comes from a full read of the upstream source (`dust.py` 608 lines, `model.py`, `baselines/backprop.py`, `prepare_data.py`) and a second pass on the experimental design. This plan supersedes v1 (`Dust-Q Handoff Implementation Plan.md`) and v2 (`Dust-Q Full-Paper Expansion.md`) as the execution baseline. v1's agent rules (section 0), claim tags, logging, no-test-tuning and per-task reports all still apply. Revise this file in place; do not append.

Tags: [paper] = Dust report; [code] = verified by two independent reads of upstream `main` on 2026-10-10 (dust.py, model.py, baselines/backprop.py, prepare_data.py; pin the commit in T0); [design] = our choice; [unverified] = must be checked.

### What changed in 3.1 (summary)

1. **New core baseline: backprop + Gaussian-smoothed STE (BP-SSTE).** This is the control reviewers will demand. It separates "Dust helps because it smooths the quantizer" from "Dust helps because it measures the real loss". Either result is publishable (section 3).
2. **Quantizer redesign for 4-bit and 2-bit activations.** Per-token absmax at A2 would round almost everything to zero. 3.1 uses MSE-calibrated clipping, an unsigned grid after ReLU², and exact integer-code matmuls (section 4.2).
3. **Attention internals are credited via a local linear surrogate, not the full quantized forward** [code]. This decides which activations we may quantize (section 2, 4.2).
4. **Upstream noise scales are hard-coded inside `DUST.step()`** [code]. A thin subclass is required for σ sweeps and tuning (section 4.1).
5. **Cost model corrected.** The minimal code reruns the *full* forward for every draw (no prefix caching), and attention is about 40–50% of the FLOPs at sequence length 2048. Budget re-derived, including a **tuning budget** that v1–v3 under-counted (section 6).
6. **One primary endpoint, pre-registered**, with everything else secondary, to avoid a multiple-comparisons critique (section 9).
7. **Minimum publishable paper defined** (about 45 L4-h), so the project still produces a paper if time or compute runs short (section 6.3).
8. Fixes: upstream `main()` requires CUDA, so CPU smoke tests use our runner. Upstream has no resume support, so our runner saves optimizer momentum and the noise RNG. A 20M budget is a trivial prefix of the 20.5M pool that upstream already encodes.

---

## 1. Feasibility verdict

**Feasible, and more robust after 3.1.** The project has a publishable outcome on every branch:

| Outcome at the primary cell (S, 1M, W1.58A4, K=1k) | Paper story |
| --- | --- |
| Dust beats BP-STE **and** BP-SSTE | Measuring the true quantized loss matters beyond smoothing; activation-space ZO is a real alternative to STE |
| Dust beats BP-STE, ties BP-SSTE | Dust's quantization advantage is implicit smoothing; analytic smoothing recovers it at about 1/K the cost. A clean mechanistic result |
| Dust ties or loses to BP-STE | Characterization: STE-bias and σ–Δ probes explain why activation-space ZO gives no edge at these bit widths |

Venue: **TMLR first** (a journal that judges evidence, not scale). Fallbacks: Neural Networks, then Neurocomputing, then Knowledge-Based Systems / Pattern Recognition. IEEE TNNLS is a stretch.

### 1.1 From scratch vs pretrained (decided: from scratch)

- **Novelty**: all quantized-ZO work (QuZO, QZO, ZeroQAT, FOGZO, ZOQO) fine-tunes pretrained models. From-scratch pretraining is the open cell.
- **Comparability**: Dust's targets (Appendix C) are from-scratch runs with its 4096-token BPE tokenizer.
- **Cost**: models are 7M/38M parameters at 1M/10M tokens. Cost scales with K (draws per step), not with the starting point. A public pretrained model (GPT-2 124M, 50k vocab) would make every draw roughly 4–10× more expensive [design].
- Checkpoints are still used for probes (our own backprop runs). Optional cell D5 tests "continue from our own fp checkpoint".

### 1.2 Problems carried over from v3 (still valid)

- P1: at A8, STE is close to harmless, so the comparison needs an activation-bit axis.
- P2: σ–Δ is a smoothing bias-variance question, not a "dead zone".
- P3: the STE gradient is not ground truth. Use a high-K Dust reference plus a directional-descent test.
- P4: upstream recipes are hard-wired to 8 blocks and K ∈ {256, 1k, 4k, 16k}.

---

## 2. Upstream code facts that shape the design [code]

**Estimator structure (`DUST.step`)**
- Direct, full-forward draws jitter outputs at these sites:
  - `attn.c_proj` + `mlp.c_proj` (jittered together as "writers"), σ=0.2;
  - `mlp.c_fc` ("hidden"), σ=0.2;
  - `transformer.wte`, σ=0.2;
  - the attention output `o@l` (pre-hook on `attn.c_proj`), σ=0.2 for the first half of the blocks and 0.4 for the deeper half.
- **σ values are literals inside `step()`**, not config.
- **Attention internals** (q, k, v, ve_gate, value_embeds) go through `local_errors`. It recomputes only that block's attention from *cached clean* q/k/v and scores Δo against the estimated error at `o@l` (a linear surrogate, with γ-decayed credit to later tokens). These credits **do not pass through any downstream quantizer except via the `o@l` error**, which is itself measured through the real forward.
- **Head**: `head_error` jitters cached logits one 256-column slab at a time and recomputes CE analytically. **Residual mixing scalars**: two-sided weight-space ES, 8 draws (16 full forwards) per step.
- **G_W** = `einsum('btd,bti->di', error, captured_input)`, where the input is captured from `args[0]` in a forward hook on every `nn.Linear`/`nn.Embedding`. Gradients are written to `.grad` and then `torch.optim.SGD.step()` runs.
- Optimizer groups: wte lr = 1000, value_embeds lr = 0.3, lambdas lr = 0.03, all else `settings.lr`.

**Cost structure**
- `direct_errors` calls `self.model(...)` in full for every chunk of `nrep` draws. **There is no prefix caching in the minimal code** (the paper's optimized code has it [paper]).
- Full forwards per step ≈ K (direct) + 16 (scalars), each over 16k tokens. Local and head draws are comparatively cheap.

**Model**
- Pre-norm, QK-norm, rotary, ReLU² MLP (4×), value embeddings on alternate blocks, logit soft-cap 15.
- wte and value_embeds are stored in **bf16** on CUDA.
- Attention is SDPA with an explicit boolean mask, which computes the full T×T even for half-window blocks. The window pattern is 'SSSL' (in L8, blocks 3 and 7 are full-window).

**Recipes**
- `recipe()`: K ∈ {256, 1024, 4096, 16384}; nrep ∈ {2, 8, 8, 16}.
- Per-block lists are 8 long; direct draws sum to K.
- Head draws = 4K and must be divisible by nrep × 16 slabs.

**Pairing**
- Upstream already pairs seeds: `make_model(seed)` and data order `seed+1` are identical for Dust and backprop. **Paired statistics are valid by construction.**

**Infrastructure**
- `main()` refuses to run without CUDA and refuses a non-empty output directory. It saves only `best.pt` (model weights): no optimizer state, no RNG, no resume. The noise generator is device-resident and seeded `seed + 3 + 7777·rank`, so our resume must save and restore its state.
- `evaluate()` hard-codes bf16 autocast (it ignores `self.dtype`), and both entry points set `float32_matmul_precision('high')` (TF32). G1's fp32 comparison must therefore use our runner's own evaluation path and a logged TF32 policy.

**Data**
- `prepare_data.py` skips the first 200,000 FineWeb (sample-10BT) documents, encodes a 20.5M-token pool (10,004 rows of 2049), and saves 1M/10M prefixes (488 / 4880 rows). A 20M budget (9,768 rows, steps=1221) is a one-line extension in our wrapper; it fits the pool with ~240 rows to spare [code].
- `train_1M.pt` is a strict prefix of `train_10M.pt`; saves are idempotent (existing files are compared, mismatches raise); `manifest.json` records sha256 of each file and of the tokenizer.
- Val/test come from documents after that pool (544 × 2048 each).

**Backprop baseline defaults**
- 1M: lr 0.175, momentum 0.98, 61 steps. 10M: lr 0.2, momentum 0.95, 610 steps. bf16 autocast (CPU falls back to fp32). Single GPU only (torchrun refused).
- Its optimizer has two groups: wte at lr 1000, everything else at the base lr. Unlike `DUST.make_optimizer`, backprop gives no special lr to `value_embeds` or the λ scalars — `backprop_ext.py` must preserve each method's own grouping, and any change must be symmetric across methods.

---

## 3. Research questions, hypotheses, baselines

### 3.1 Methods compared (same model, data order, init, optimizer family)
| Code | Method | Role |
| --- | --- | --- |
| BP | backprop, fp | reference |
| BP-STE | backprop, quantized forward, identity STE on activation and weight quantizers | standard QAT baseline |
| **BP-SSTE** | backprop, quantized forward, activation-quantizer backward = derivative of the **Gaussian-smoothed** quantizer E[q(x+σε)] with σ matched to Dust's per-site σ; weights use identity STE | **control for "Dust = smoothing"** |
| Dust-K | Dust with the same quantized forward; weights updated on latent fp32 with the same rule | method under study |
| (opt.) BP-LSQ | learnable step size, backprop only | "strongest STE" reference (Tier D) |
| (opt.) ZO-W | MeZO-style weight-space ZO, matched forward count | RQ5 of v2 (Tier D) |

The BP-SSTE backward is cheap: per element, Σ_k (1/σ)·φ((x − b_k)/σ)·Δ over the grid boundaries b_k (truncated to the nearest ±3 boundaries), with clamp handled the same way [design].

### 3.2 Questions and pre-registered hypotheses (commit in `docs/HYPOTHESES.md` before Tier B)

- **RQ1**: how does Dust − BP-STE change as activation bits drop (fp → A8 → A4 → A2)?
  - H1: at A8 the gap equals the fp gap within CI.
  - **H2 (primary)**: at A4, the interaction I = (Dust − BP-STE)@A4 − (Dust − BP)@fp is < 0.
- **RQ2**: is any Dust advantage explained by smoothing?
  - H3: Dust − BP-SSTE at A4 is < 0 (Dust beats analytic smoothing). Exploratory with a pre-stated direction.
- **RQ3**: scaling with K (256 → 4k), size (7M/38M) and tokens (1M/10M). Cosine law c_max and c against bits (H4: c_max falls and c rises as bits drop).
- **RQ4**: σ–Δ mechanism. H5: at A4/A2 the optimal σ* is proportional to the median Δ with a layer-type constant; at A8, σ* ≈ upstream σ.
- **RQ5**: STE-bias probe. Does cos(STE, Dust-16k) per bit width predict the RQ1 gap?

Framing caveat (must appear in the paper): the latent-weight update is STE-like for every method. Dust removes STE only from the **activation credit path**, and for attention internals only up to the attention output (the local surrogate, section 2).

---

## 4. Implementation design (no upstream edits)

```
third_party/dust/        pinned submodule, unmodified
dustq/
  quant.py               quantizers (fp32 math), STE / SSTE autograd functions, code-matmul helper
  qmodel.py              build upstream GPT via make_model, swap Linear->QLinear, attach act-quant pre-hooks
  dust_ext.py            DUSTQ(DUST): configurable sigmas, flip-rate counters, dtype override, state_dict for resume
  recipes.py             recipe(tokens, K, n_layer) generalising upstream; L8 path byte-equal to upstream
  backprop_ext.py        BP / BP-STE / BP-SSTE training step (reuses upstream make_optimizer logic)
  probes.py              cosine ladders, split-half reference, directional descent, flip rate, STE-bias
  runner.py              one entry point for all methods; resumable; CPU-capable for smoke tests
  data.py                calls upstream prepare_data; optional 20M prefix from the same pool
scripts/ queue.py benchmark.py noise_sweep.py aggregate.py make_figures.py
configs/ results/ docs/ tests/
```

### 4.1 `DUSTQ(DUST)`: the only behavioural subclass
- Overrides `step()` with an identical copy except that σ values come from config (`sigma_writer`, `sigma_hidden`, `sigma_emb`, `sigma_hub_shallow`, `sigma_hub_deep`, `sigma_head`, `sigma_local`, `sigma_scalar`).
- Unit test: with default σ, a `DUSTQ.step` and an upstream `DUST.step` from the same seed produce **bitwise-identical** gradients on a tiny model.
- `self.dtype` override for fp32 runs. Note that wte and value_embeds stay bf16 unless we cast them too; G1 casts them.
- `state_dict()` / `load_state_dict()` cover the noise generator, so a resumed run continues the same noise stream.
- Optional counters: per-site flip rate (the fraction of activation codes that differ between the clean and jittered forward).

### 4.2 Quantized forward
**Where activations are quantized** (inputs of quantized Linears only):
- c_q / c_k / c_v input (norm(x), shared);
- attn.c_proj input (the attention output o);
- mlp.c_fc input (norm(x));
- mlp.c_proj input (ReLU² output: **unsigned grid**).

**Not quantized**: q/k after rotary and norm, attention probabilities, ve_gate, embeddings, value embeddings, LM head and its input, norms, and the λ scalars. This keeps `local_errors` and `head_error` exact for what they assume (section 2) and matches BitNet practice.

**Activation quantizer at b bits** [design]:
- Per-token scale `s = α_b · mean|x|`, where α_b is a **fixed clip multiplier chosen once** by minimising quantization MSE on activations of a backprop fp checkpoint (T5).
- The same α_b is used for both methods and is never learned. A8 may keep absmax.
- Signed grid: codes ∈ [−2^(b−1), 2^(b−1)−1]. Unsigned grid for the ReLU² input: codes ∈ [0, 2^b−1].
- At A2 the result is 4 levels (signed: {−2,−1,0,1}·s; unsigned: {0,1,2,3}·s). Per-token absmax at A2 would map most entries to 0 under ReLU²'s heavy tail, so it is not used.

**Weights**: ternary absmean (γ = mean|W| per tensor) or int8 absmax per tensor.

**Exact arithmetic**:
- Quantizer math runs in fp32. QLinear multiplies **integer codes** (exact in bf16 for |code| ≤ 256, fp32 accumulation) and rescales by s_x·γ afterwards in fp32. bf16 rounding cannot move grid points, and the "zero-noise Dust forward = quantized forward" test becomes bit-exact.
- The pre-hook passes codes and scale to QLinear via a module attribute and returns the dequantized x as `args[0]`, so upstream's capture gives G_W the real quantized input.

**Hook ordering (critical)**: upstream registers a pre-hook on `attn.c_proj` that adds `o@l` jitter. Our quant pre-hooks must be registered **after `DUSTQ(...)` is constructed**, so the order is noise → quantize. Tests 3–4 in T5 verify this.

### 4.3 Recipes
- L8 = upstream (asserted equal).
- L4: merge adjacent block pairs → writers [70,26,10,4], hubs [12,6,4,4]. Divisible by nrep at all four K values (checked by test).
- Validate L4 against a small allocation search with the cosine method. Accept if the merged recipe is within 5% cosine of the searched one.
- K=64 and other non-preset K values: Tier D only.
- Check the head count for L4/d256 against the paper's appendix [unverified; default n_head=8 gives head_dim 32].

### 4.4 Runner and queue
- `runner.py` writes `results/<run_id>/` with config, metrics.jsonl, `ckpt.pt` (model + optimizer + DUSTQ RNG + step + data cursor) every 10 min, `best.pt`, and `result.json`.
- `queue.py` reads a run list, **skips completed runs, resumes partial ones**, and appends to `results/runs.csv`. Every Colab session just runs `queue.py` (idempotent).
- Data and results live on Drive. The data files total about 50 MB (int32: train_10M ≈ 38 MiB, train_1M / val / test ≈ 4 MiB each).
- T1 tests running **2 S-size jobs concurrently on one L4**, since the S model will underuse the GPU.

---

## 5. Probes (run before long training)

All probes run on our own backprop(-STE) checkpoints at 1M tokens, per precision.

1. **Reference gradient**: Dust at K=16k, split into two independent 8k halves. The cosine between the halves estimates the reference's own noise, which is used to correct cosines to the reference (disattenuation).
2. **Cosine ladders**: K ∈ {256, 1k, 4k} per layer type, against the reference and against STE. Fit cos(K) = c_max/√(1+c/K).
3. **STE-bias probe (RQ5)**: cos(STE, ref) and cos(SSTE, ref) by bit width and layer type. **This runs first.** If cos(STE, ref) ≈ cos(fp-BP, fp-ref) at A4, H2 is unlikely, and we know before spending the Tier B budget.
4. **Directional descent**: for ĝ ∈ {STE, SSTE, Dust-K, ref}, line-search ε on a log grid and record min_ε L_q(θ − ε ĝ/‖ĝ‖) − L_q(θ), on the training batch and on one held-out batch. L_q is the real quantized loss, so this is the method-neutral arbiter.
5. **σ–Δ sweep (RQ4)**: per site group and bit width, σ ∈ {1/8…16}× median Δ at K ∈ {256, 1k}. Record cosine to ref, flip rate, mean |reward| and directional descent. Fit σ*. **Run on S only**; spot-check on M at A4.
6. (could) Sharpness as in v1 section 9. It is tangential to the quantization story, so it was demoted.

---

## 6. Experiment matrix and budget

### 6.1 Cost model [design: T1 replaces it]
FLOPs per full forward over 16k tokens:
- **M (L8/d512)**: ≈ 0.9 TFLOP matmul + 0.55 TFLOP attention ≈ 1.45 TFLOP.
- **S (L4/d256)**: ≈ 0.14 + 0.14 ≈ 0.28 TFLOP.

At about K+16 forwards per step, an assumed ~40 TFLOPS achieved on L4 (M) and ~25 TFLOPS (S, poorer utilization), plus about 30% for fake-quant overhead:

| | K=256 | K=1k | K=4k |
| --- | --- | --- | --- |
| M s/step | ~13 | ~50 | ~200 |
| S s/step | ~4 | ~15 | ~60 |
| M 1M run (61 steps) | 0.2 h | 0.85 h | 3.4 h |
| S 1M run | 0.07 h | 0.25 h | 1.0 h |
| S 10M run (610 steps) | 0.7 h | 2.5 h | 10 h (excluded) |
| M 10M run | 2.2 h | 8.5 h (excluded) | – |

Backprop runs take minutes. **If T1 measures more than 1.5× these figures, re-cut using section 6.4.**

### 6.2 Matrix
Precisions:
- P0 fp
- P1 W8A8 (secondary)
- P2 W1.58A8
- **P3 W1.58A4 (primary)**
- P4 W1.58A2 (gated by G2b)

"BPx" means BP for P0, and BP-STE + BP-SSTE for P1–P4. All cells use 3 seeds unless noted.

| Tier | Cells | Dust runs | Est. L4-h | Priority |
| --- | --- | --- | --- | --- |
| R | M/1M/P0: BP + Dust K∈{256,1k,4k}. S/10M/P0: BP + Dust K256 (derived recipe vs paper's 5.362) | 12 | ~16 | must |
| P | probes 1–5 (S all precisions; M at P0/P3) | – | ~6 | must |
| Tune | section 6.5 | – | ~10 | must |
| B-S | S/1M, P0–P4 × {BPx, K256, K1k, K4k} | 45 | ~20 | must |
| B-M | M/1M, P2, P3 × {BPx, K256, K1k} (P0 reused from R) | 12 | ~7 | must |
| A | S/10M, P0, P3 × {BPx, K256, K1k} (+P2 if budget allows) | 12 (+6) | ~20 (+10) | must |
| D1 | M/10M, P0, P3 × {BPx, K256} | 6 | ~14 | should (gated on T1) |
| D2 | +2 seeds on the primary cell and its fp counterpart (Dust K1k, BP, BP-STE, BP-SSTE) | 4 | ~2 | **approved** |
| D3 | BP-LSQ, ZO-W, BP-noise-injection at S/1M/P3 | 3 | ~2 | could |
| D4 | discrete-update ternary pilot; K=64 | 3–6 | ~4 | could |
| D5 | continue from our fp checkpoint → P3, Dust K256 vs BP-STE/SSTE | 3 | ~1 | could |

**Core (R + P + Tune + B-S + B-M + A) ≈ 80 L4-h**, about 135 CU at 1.7 CU/h [unverified]. With D1 and D2 it is about 96 L4-h (≈165 CU). That is one 200-CU month if the estimates hold, two months with slack.

### 6.3 Minimum publishable paper (if time or compute runs short)
R (M/1M part) + P + Tune (S only) + B-S, with BP-SSTE included: **≈ 45 L4-h**. That supports H1/H2/H3, the probes and the K axis, which is enough for a TMLR submission scoped as "a 1M-token study". Tier A and B-M then become the revision-stage additions.

### 6.4 Cut order if over budget
D4 → D3 → D5 → D1 → A's P2 → B-M K1k → P1 (W8A8). **Never cut**: the primary cell, its fp counterpart, BP-SSTE, or the probes.

### 6.5 Tuning protocol (validation only; fixed in advance)
- **Noise scales**: from probe 5 (no training). For A8 and fp, use the upstream σ.
- **Dust LR**: a 3-point grid {0.6, 1, 1.5}× the upstream default, 1 seed, at K=256 and K=1k per precision, on S/1M. K=4k inherits K=1k's multiplier (upstream LR is nearly flat in K [code]).
- **M and 10M**: inherit S/1M multipliers on top of the upstream defaults for that budget. Verify with a 2-point check at K=256 only.
- **Backprop variants**: a 5-point LR grid × 2 momenta per (size, budget, precision). They are cheap.
- Tuning runs use seeds 100+ and **never count as reported seeds**.
- Asymmetry: backprop gets a denser grid. This biases against our hypotheses and is stated in the paper.

---

## 7. Phased task list
Each task ends with `docs/reports/T*.md`.

### Phase 0: recon and setup (week 1, no GPU)
- **T0**:
  - Pin upstream. Write `docs/REPO_MAP.md` from section 2, adding line references at the pinned commit.
  - Extract Appendix C–G into `docs/paper_tables.md`, including the L4/d256 configuration (heads, recipe, and whether σ differs by size).
- **T0-lit**: full texts of QuZO, QZO, ZeroQAT, HQZO, FOGZO, ZOQO; node perturbation with decorrelation; BitNet b1.58 and BitNet a4.8 (4-bit activations); DSQ and other smoothed or noisy quantizer-gradient work (prior art for BP-SSTE). At least 15 verified refs in `docs/related_work.md`.
- **T0-hyp**: commit `docs/HYPOTHESES.md` (section 3.2) and the analysis plan (section 9).
- Acceptance: on **CPU via our runner**, a tiny model (L2/d64, seq 128) takes 3 DUSTQ steps and 3 BP-STE steps; the DUSTQ-vs-DUST bitwise test passes.

### Phase 1: infrastructure and reproduction (weeks 1–2)
- **T1**:
  - Runner, queue, resume test: kill mid-run, then resume; the final loss matches an uninterrupted run exactly (same RNG).
  - `benchmark.py`: s/step for S and M at K ∈ {256, 1k, 4k}, with and without fake-quant; peak memory at nrep 8/16; 2-concurrent-S throughput; CU/h from Resources.
  - **Re-cut section 6.**
- **T2**: upstream data prep with `--tokens both`. Hash-check that train, val and test are disjoint. Record the manifest.
- **T3 / G0**: Tier R. Pass: M/1M within 0.05 of Appendix C at each K, with the same K ordering. S/10M/K256 within 0.08 of 5.362.
- **T4 / G1**: fp32 (dtype override plus fp32 embeddings) vs bf16 on M/1M/K256 at P0 and P3. Final loss and single-batch cosine within seed noise.

### Phase 2: quantization and mechanism (weeks 2–3)
- **T5**: quant.py, qmodel.py, recipes.py, backprop_ext.py. Tests:
  1. weight codes are in range;
  2. the activation quantizer gives the correct levels for b ∈ {8,4,2}, signed and unsigned;
  3. **noise-then-quantize order at `o@l`** (compare against a manual forward);
  4. captured input = dequantized quantized input;
  5. zero-noise DUSTQ forward = quantized forward, bit-exact;
  6. recipes L8 = upstream; L4 divisibility holds;
  7. SSTE backward matches a finite-difference derivative of the Monte-Carlo smoothed quantizer;
  8. code-matmul QLinear vs an fp64 reference within 1e-3 relative.

  Calibrate α_b by MSE on a BP fp checkpoint (S and M) and freeze it in configs. Smoke test: 100 steps per precision per method, with no NaN.
- **T6 / G2**: probes 1–5. **G2a**: at P3, cos(Dust-1k, ref) rises with K on MLP and writer layers, and directional descent along Dust-1k is negative. **G2b**: the same at P4.
- **T7**: tuning per section 6.5.

### Phase 3: main runs (weeks 3–6)
- **T8**: B-S, then B-M, then **G3**. Post an arXiv preprint after G3.
- **T9**: Tier A. **T10**: analysis (cosine ladders on Tier A checkpoints, gap vs bits, probe-gap correlation, loss vs FLOPs).

### Phase 4: extensions and writing (weeks 6–10)
- **T11**: D2 (approved), D1 if T1 allows, D3–D5 after G3.
- **T12**: `make_figures.py` regenerates every figure and table from `results/runs.csv`. Write the TMLR draft. Release the code.

---

## 8. Gates
| Gate | After | Pass | If it fails |
| --- | --- | --- | --- |
| G0 | T3 | M/1M within 0.05, same K ordering | Debug dtype/data/recipe. If still off, report a reproduction gap and continue with relative comparisons only (all methods share our pipeline) |
| G1 | T4 | bf16 ≈ fp32 at P0 and P3 | Run quantized cells in fp32 (about 2× cost); apply the 6.4 cuts |
| G2a | T6 | Dust signal at P3 (section 7) | Probe-only paper (STE-bias + σ–Δ + directional descent); still TMLR-scoped |
| G2b | T6 | Dust signal at P4 | P4 becomes probe-only |
| G3 | T8 | Primary contrast estimated with CI half-width < 0.03 nats, or a clear failure map | Add D2 seeds early; otherwise write a workshop characterization paper |
| G4 | D4 | Discrete pilot trains, or its failure is attributable to section 5 | Drop it |

At each gate, write a one-paragraph recommendation for the owner before continuing.

---

## 9. Statistics and analysis plan (pre-registered)
- **Primary endpoint**: interaction I = [Dust-1k − BP-STE]@P3 − [Dust-1k − BP]@P0 on S/1M, test loss at best validation. Paired by seed. 95% CI via paired bootstrap (10k resamples). **Claim H2 only if the CI excludes 0.** With D2, n = 5 seeds.
- **Key secondary**: Dust-1k − BP-SSTE @P3 (H3), same procedure.
- **Everything else is descriptive or exploratory**: other K, sizes, tokens, bit widths. Report CIs and no significance language. Trend over bits is reported as a fitted slope with CI.
- Divergence counts as a reported outcome. Diverged runs are scored at the best validation reached before divergence and flagged.
- No run is ever dropped. Deviations from this plan are listed in the paper's appendix.

---

## 10. Paper skeleton (TMLR)
1. **Intro**: STE is a surrogate. Activation-space ZO measures the real quantized loss. Does that matter, and is it more than smoothing?
2. **Background and related work**: Dust; node perturbation; BitNet b1.58 / a4.8; STE bias; smoothed and noisy quantizer gradients; quantized ZO (QuZO, QZO, ZeroQAT, FOGZO, ZOQO), placed on two axes (weight vs activation space; fine-tuning vs from-scratch).
3. **Method**: quantizers and placement, hook ordering, BP-SSTE, the exact STE caveat (including the local-surrogate scope for attention internals).
4. **Probes**: Fig 1 STE/SSTE-bias vs bits; Fig 2 σ–Δ sweep and flip rate; Fig 3 cosine ladders and law fits; Fig 4 directional descent.
5. **Training**: Fig 5 loss vs K per precision with BP/BP-STE/BP-SSTE lines; **Fig 6 primary interaction (headline)**; Fig 7 size/token axes; Table 1 main results; Table 2 σ*/Δ.
6. **Analysis**: does the probe predict the gap; compute cost stated honestly [paper].
7. **Limitations**: ≤38M parameters, ≤10M tokens, one epoch; STE-like latent updates; fixed (non-learned) quantizer scales; Dust compute cost; minimal-code reproduction gap if any.

---

## 11. Risks (3.1)
- **Hook-order or capture bug** silently invalidates quantized Dust. Mitigation: T5 tests 3–5. This is the top engineering risk.
- **BP-SSTE matches Dust**: not a failure. It is the "smoothing explains it" outcome (section 1).
- **A4 is too benign** (no STE bias): detected cheaply by probe 3 before Tier B. Then P4 becomes primary, if G2b passes.
- **Derived S recipe or LR transfer underperforms**: mitigated by the G0 S-check and tuning. M remains the anchor.
- **Minimal-code reproduction gap** vs the paper's optimized code: all comparisons are internal and paired, so a uniform offset does not affect our claims. Report it.
- **CU rate / session limits**: the idempotent queue plus 10-minute checkpoints make any session productive. The minimum paper (6.3) is about 45 L4-h.
- **Scooping** (Dust authors, QuZO/FOGZO groups): preprint after G3; a monthly arXiv check for "Dust" + quantization / ternary / non-differentiable.

---

## 12. Owner decisions (recorded 2026-10-10)
1. Activation-bit axis as headline: **adopted** (agent's call, delegated by owner).
2. M = L8/d512 anchor; S = L4/d256 with derived recipe; K=64 out of core: **approved**.
3. 200 CU/month with L4 access: **confirmed**. CU/h still to be logged in T1.
4. Venue: any suitable journal, so TMLR first (section 1).
5. Extensions: **D2 approved**; D1 gated on T1; D3–D5 decided after G3.
6. Train from scratch: **decided** (section 1.1).

Timeline [design]: about 8–10 weeks part-time. The minimum paper (6.3) is reachable in about 5 weeks. The owner's time, not compute, is the binding constraint.
