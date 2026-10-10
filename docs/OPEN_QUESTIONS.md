# Open Questions

Seeded 2026-10-10 during the plan v2 expansion. Updated by any agent who resolves or adds an item. Format: status, context, who/when.

## Owner to confirm

1. [unverified] Kaggle quota: "45 + 30 T4 hours" - period, and is it two separate quotas?
2. [resolved 2026-10-10, owner] 200 CU/month tier has L4 access. Still open: the CU/h actually observed in Resources (T1).
3. [resolved 2026-10-10, owner] Venue: any suitable journal -> TMLR first, fallbacks in Plan v3 section 1.

## T0 must resolve (upstream code, github.com/qlabs-eng/dust)

4. [resolved 2026-10-10, code read] bf16 is chosen in `DUST.__init__` (`bfloat16 if cuda else float32`) but is an instance attribute (`self.dtype`), so it can be overridden after construction without editing upstream. fp32 on T4 possible (slow). See Plan v3 section 2.
5. [resolved 2026-10-10, code read] Not out of the box: `recipe()` only accepts K in {256,1024,4096,16384}; per-block allocation lists are hard-wired to 8 blocks; K=64 gives `scale=0`. Plan v3 adds `dustq/recipes.py` (L4 via block-pair merging) and drops K=64 from the core.
6. [resolved 2026-10-10, code read] No: `--tokens` choices are {1M, 10M, both} (488/4880 sequences). 20M needs our own data script (Plan v3 `dustq/data.py`).
7. [resolved 2026-10-11] Upstream pinned and vendored: commit b20f7c03eac630b6441ba6f254128c1761dc45ad, MIT, vendored at third_party/dust (UPSTREAM_COMMIT records the pin). The stray root prepare_data.py was verified content-identical to the pinned copy (line endings only) and is gitignored.
8. [resolved 2026-10-10, code read] No flash-attn: attention uses `F.scaled_dot_product_attention` with an explicit mask. Open sub-item: memory ceiling at nrep=8/16 with the explicit mask (T1).
8b. [resolved 2026-10-10, full dust.py read] Update path, head/local/scalar estimators, hard-coded sigmas and cost structure documented in Plan v3.1 section 2. Remaining: the pre-hook ordering must be verified by test (Plan v3.1 T5 tests 3-4), and the L4/d256 head count and recipe must be taken from the paper appendix (T0).

## T1 must resolve (benchmarks)

9. [unverified] Single-L4 memory ceiling: does K=4k fit without OOM (upstream used 4 GPUs at K=4k for speed [paper])? Same for the 38M model.
10. [unverified] Measured L4 CU burn rate; replaces all hour estimates in Plan v2 section 6.

## T0 must resolve (novelty)

11. [first pass clean 2026-10-10] Full-text reads of QuZO (arXiv:2502.12346), QZO (arXiv:2505.13430), ZeroQAT (arXiv:2509.00031), HQZO: confirm none claims activation-space perturbation under a quantized forward.
12. [unverified] Additional searches still to run: forward gradients + quantization; node perturbation + low-bit; BitNet b1.58 exact identifier; ES-trained binarized/ternary nets; "Dust" citations and follow-ups appearing after Oct 2026.

## Second independent code read (2026-10-10, later session)

Purpose: confirm Plan v3.1's [code] claims before execution. Sources: upstream dust.py, model.py, baselines/backprop.py fetched from `main`; local prepare_data.py read in full. Result: every checked claim holds. Facts recorded below go into REPO_MAP.md at T0.

13. [verified] Estimator claims: sigma literals confirmed (writers/hidden/wte 0.2; o@l hubs 0.2 shallow blocks / 0.4 deep; head 0.05; local 0.05; scalars 0.03 with 8 two-sided draws = 16 full forwards per step); inputs captured as `args[0].float()` in the output hook, so pre-hook replacements are visible to G_W; embedding gradients via index_add; non-finite estimates raise; recipe `scale = K // 256`, so K=64 yields zero draws everywhere (embedding included).
14. [verified] Backprop baseline: 1M lr 0.175 momentum 0.98 (61 steps); 10M lr 0.2 momentum 0.95 (610 steps); bf16 autocast on CUDA, fp32 on CPU; single GPU only. Optimizer-group asymmetry vs DUST: backprop has {wte: 1000, rest: base lr} with no special lr for value_embeds or lambdas. backprop_ext must preserve each method's own grouping.
15. [verified] Data pipeline: 200,000-document skip; pool = 10,004 rows of 2049 (~20.5M tokens); 1M/10M = 488/4880 rows; val/test 544 rows each from documents after the pool; idempotent saves (mismatch raises FileExistsError); manifest sha256s (files + tokenizer); total ~50 MB (v3.1's ~100 MB corrected). 20M budget end-to-end: 9,768 rows fit the pool; needs steps=1221 and a wrapper recipe.
16. [verified] Misc for REPO_MAP: window pattern 'SSSL' (L8: blocks 3 and 7 full-window); make_model re-inits both c_proj weights to ±1.5·sqrt(3/fan_in) uniform, lm_head std 0.001, x0_lambdas 0.1; evaluate() hard-codes bf16 autocast (ignores self.dtype); float32_matmul_precision('high') in both entry points; --steps smoke flag in both; both mains refuse non-empty output dirs; noise generator device-resident, seeded seed+3+7777·rank.
17. [verified, de-risks T5 tests 3-4] v3.1's quantizer placement (4.2) is consistent with the code: local_errors recomputes block attention from cached c_q/c_k/c_v outputs (rotary then QK-norm applied identically to the real forward) and never touches our activation quantizers; head_error works on cached logits; quantizing the inputs of c_q/c_k/c_v, attn.c_proj, mlp.c_fc, mlp.c_proj is exactly the compatible set. The noise-then-quantize pre-hook order is feasible: upstream's o@l pre-hook returns modified args, so a later-registered pre-hook sees the noisy input and the output-hook capture then stores the quantized dequantized value.
18. [open, T4] TF32 policy for fp32 comparisons (upstream enables TF32 via matmul precision 'high'): decide and log alongside G1.

## Appendix extraction (2026-10-11, docs/paper_tables.md)

19. [resolved 2026-10-11] L4/d256 head count: 4 heads — derived from Appendix A's "same head dimension" (base 8 heads at width 512, head dim 64; 256/64 = 4). Not printed as a number; treat as derived. Plan v3.1 4.3 and qmodel updated to n_head=4 for S.
20. [resolved 2026-10-11] Noise scales by model size: the appendix never states per-size noise scales; they are per layer type (0.2; deeper-half hubs 0.4; head/local 0.05; scalars 0.03), "held fixed across cells", re-tuned in training only at the 10M/20M 16k cells. Using the upstream literals for S and quantized variants is therefore protocol-consistent; probe 5 still re-tunes them for quantized precisions.
21. [note, T3] The S/10M reproduction target (5.362) comes from main-post Figure 4, whose size-study protocol predates Section 3.1 (Appendix F: token-embedding lr 0.3/base, key 4x / value 8x rates). The vendored code implements the MAIN protocol (Table 6 main-comparison columns match the code exactly, incl. wte lr 1000 both methods). Expect S runs on code defaults to land 0.02–0.05 below Figure 4 values (the M gap is 4.989 vs 5.015 at 10M); G0's 0.08 tolerance covers it, and all claims are internal and paired.
22. [resolved 2026-10-11] Appendix C main-comparison targets now pinned in Plan v3.1 T3: M/1M Dust 256/1k/4k = 6.067/5.934/5.911, backprop 5.959; M/10M Dust 256/1k = 5.231/5.111, backprop 4.989. Selected SGD (lr, momentum) winners are NOT printed for any cell — the minimal code's DEFAULT_CONFIGS are our reproduction's tuned values (its README says it preserves tuned defaults).
