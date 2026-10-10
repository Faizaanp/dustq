# Dust-Q Full-Paper Expansion (Plan v2)

> **Status (2026-10-10, later the same day): superseded as the execution baseline by `Dust-Q Implementation Plan v3.md` (revision 3.1).** v3.1 absorbed this document's surviving value — the TMLR-first venue call, the novelty positioning (section 2 below), the sigma-Delta study (now probe 5 / RQ4, reframed from "dead zone" to smoothing bias-variance), the discrete-update variant (D4/G4) and the weight-space-ZO baseline (D3) — and corrected three of its errors: the budget table assumed the paper's prefix caching while the minimal code reruns the full forward per draw; W1.58A8 is no longer the headline (STE is near-harmless at A8; the primary endpoint is the W1.58A4 interaction); and K=64 cannot run upstream (recipes scale as K/256, so K=64 yields zero draws). Sections 1-2 remain useful context. Do not execute from this file.

Companion to `Dust-Q Handoff Implementation Plan.md` (v1, 2026-10-10). v1 remains the execution baseline; this document upgrades the target from a workshop paper to a full research paper, re-plans compute around Colab L4, and adds the mechanistic study that makes a full paper defensible. All v1 rules apply: claim tags [paper]/[design]/[unverified], no invented numbers, no test-set tuning, log every run, per-task reports in docs/reports/.

Status: planning. Evidence gathered 2026-10-10 (upstream repo and README inspection; novelty first pass). Nothing here overrides a gate decision.

## 1. Verdict: is a full research paper possible?

**Yes, conditionally.** The niche is open as of 2026-10-10 [verified by search; T0 must run the deeper pass], the base method is real and reproducible against published targets, and the compute fits free tiers. Venue tiers, in order of confidence:

| Tier | Venue | Confidence | What it takes |
| --- | --- | --- | --- |
| 1 | Workshop paper | High | v1 plan as written (Tiers B/A) |
| 2 | **TMLR (recommended primary)** | Medium-high | v1 + section 4 study + Tier D1 + 5 seeds on headline cells |
| 3 | Q2+ journal (Neurocomputing, Pattern Recognition, Knowledge-Based Systems) | Medium | Tier 2 + D2 (38M at 10M tokens) + discrete-update or ZO-baseline section |
| 4 | NeurIPS/ICLR/ICML main track | Low (stretch) | Strong positive G2/G3 results and a crisp mechanistic story; model scale (38M max, 10M tokens) is the main reviewer risk |

Conditions (all four must hold):

1. G0 passes: we reproduce upstream Dust within v1's tolerance.
2. G2 passes: quantized cosines rise with K.
3. T0's deep novelty pass confirms the positioning in section 2.
4. T1's measured CU burn rate leaves room for Tier D (section 6).

Negative or mixed results still support tiers 2-3 if characterized: a clean failure map of where activation-space credit assignment breaks under quantization, plus the section 4 mechanism, is a publishable full study on its own.

## 2. Novelty landscape (first pass 2026-10-10; T0 must finalize)

Verified today by abstract and paper inspection:

| Work | Perturbation space | Task | Precision | STE claim |
| --- | --- | --- | --- | --- |
| QuZO (arXiv:2502.12346, EMNLP 2025 main) | weights (MeZO-style) | fine-tuning | 4/8-bit forwards | yes: "avoid the error-prone low-precision straight-through estimator"; stochastic rounding against quantization bias |
| QZO (arXiv:2505.13430, ICLR 2026) | weights | fine-tuning | low-bit | yes |
| ZeroQAT (arXiv:2509.00031) | weights (MeZO family) | QAT/fine-tuning of pretrained LLMs | weight and activation quant | yes |
| HQZO (Springer chapter) | weights | fine-tuning | quantized | [unverified details] |
| Allaire et al. 2025 (ICPRAM) | weights | pretraining from scratch | fp | n/a; documents the difficulty |
| KronZO (Allaire et al. 2026) | weights, Kronecker structure | pretraining from scratch | fp | n/a |
| Dust (Q Labs 2026) [paper] | activations (per token) | pretraining from scratch | fp | implicit (no backward pass) |
| Zoop (ICML 2025 workshop) | outputs | fine-tuning | fp | n/a |

The unclaimed cell: **activation-space perturbation + low-bit quantized forward + pretraining from scratch**. No hit found on 2026-10-10 [unverified: search depth; T0 must settle].

Claim discipline:

- We may NOT claim "first ZO training of quantized models" (the QuZO family owns that, for fine-tuning), nor "ZO avoids STE" as a novel property (QuZO states it).
- We MAY claim (pending T0 and G2): the first study of the only ZO method competitive at from-scratch pretraining [paper] under a quantized forward pass; the first characterization of the noise-versus-quantization-grid interaction in activation space (QuZO's stochastic rounding is the weight-space counterpart and must be cited); population-efficiency of quantized models trained from scratch.
- Positioning sentence for the paper: weight-space ZO under quantization exists but inherits weight-space ZO's known weakness at from-scratch pretraining [paper]; activation-space perturbation is the method that made from-scratch ZO pretraining competitive; whether it survives a low-bit forward pass is the open question we answer.

Scooping: Dust's conclusion explicitly names looped and non-differentiable architectures as future work [paper]. The Dust authors are the most likely scoopers; the QuZO group is second. Mitigation: early arXiv after Tier B (v1 already mandates this), narrow claims, and the section 4 characterization as our defensible core.

## 3. Research questions v2

- RQ1 (reframed): at a matched token budget, does Dust - which measures the true loss response of the actual quantized forward - train int8 and ternary transformers from scratch better than backprop+STE, which follows a smooth surrogate gradient? (Framing caveat of v1 section 1 applies unchanged: the latent-weight update is STE-like; only the credit path differs.)
- RQ2 (extended): how does the gap change with K, model size (2M / 7M / 38M) and token budget (1M / 10M)?
- RQ3 (unchanged): do quantized Dust gradients still follow the cosine-vs-population law, and do Dust and backprop reach minima of different sharpness?
- RQ4 (new): how does the noise scale sigma interact with the activation quantization step Delta? Is there a dead zone below some sigma_min, a usable range, and does the tuned sigma/Delta ratio transfer across layer types, precisions and sizes?
- RQ5 (optional, budget-gated): does activation-space perturbation keep its population-efficiency edge over weight-space ZO when the forward pass is quantized? One cell: L4/d256, ternary, 1M tokens, MeZO/SPSA-style weight-space ZO with the same quantized forward, 3 seeds.

## 4. Mechanistic core: noise scale vs quantization step (sigma vs Delta)

Why it matters [design, derived from v1's quantizer definitions]: with per-token absmax int8 activations, the grid step at token t is Delta_t = max(|x_t|)/127. Dust injects noise on linear outputs, upstream of the next activation quantizer. For sigma much smaller than Delta, most draws leave every affected rounding decision unchanged: the quantized forward is identical, c_s = 0, and the draw contributes noise, not signal. The fraction of "live" draws p(sigma/Delta) is measurable on one batch without any training. QuZO observed the weight-space analogue and answered it with stochastic rounding [verified]; the activation-space analogue at pretraining scale is uncharacterized.

Expected shape [design, to be tested, not assumed]: cosine rises from zero with sigma, peaks at some sigma* ~ c * Delta, then falls as the linear-reward model breaks and interference grows. If ternary or int8 Dust fails at low K, this mechanism is the candidate explanation - which turns a failure into a result.

Deliverables (new task T6.5, cheap, no training):

- scripts/noise_sweep.py: on one fixed batch, per layer type, sweep sigma over a grid spanning Delta (0.125x to 16x in 2x steps) at K in {64, 256, 1k}; record (a) cosine to the backprop+STE gradient, (b) live-draw fraction p, (c) mean |c_s|.
- Output: sigma_min per layer type and precision, the fitted peak sigma*, and a tuning rule for T7: anchor noise-scale grids at the measured peak rather than at fp-tuned values.
- Paper: Figure 5 (cosine vs sigma/Delta per layer type, fp reference line) and the tuning rule stated explicitly.

## 5. Compute re-plan around Colab L4 (bf16 resolved)

The bf16 constraint is real and stated by upstream: "Requires Linux, Python 3.10+, and an NVIDIA GPU with bfloat16 support" (torch 2.10.0, cu128) [verified in README]. Resolution: Colab L4 becomes the primary training GPU.

- L4 (Ada Lovelace, sm_89): bf16 native; matches upstream's required path exactly. Training in upstream's tested dtype removes a whole class of reproduction risk (G0) that fighting fp16-on-T4 would have created.
- Kaggle T4 (Turing, no bf16): demoted to data prep and CPU work, plus quick debugging only if upstream can run fp32 [unverified: T0 must check whether bf16 is hardcoded; if it is, do not patch upstream for T4 - it is not on the critical path].
- The v1 dtype gate is rewritten: G1 becomes "bf16 vs fp32 losses and cosines within noise on L4" (upstream-native vs reference). fp16-with-loss-scaling experiments are dropped unless a T4 training fallback ever becomes necessary.
- Memory checks for T1 [unverified]: L4 has 24 GB. Verify K=4k on a single L4 (upstream paper configs used 4 GPUs at K=4k for speed [paper]; single-GPU OOM behavior must be checked, not assumed) and the 38M model at K=256/1k.
- CU budget [unverified, must be measured in T1 and logged per v1 rules]: at ~1.7 CU/h for L4, 200 CU/month is roughly 115 L4-hours. Confirm which Colab tier exposes L4.
- Session limits unchanged: jobs under 8 hours, checkpoint every 10 minutes to Drive, all runs resumable (v1 T1).

## 6. Experiment matrix v2

| Tier | Content | Runs | Est. T4-eq hours [design] |
| --- | --- | --- | --- |
| B (1M tokens) | as v1: L2/d128 + L4/d256; fp/int8/ternary; backprop 3 seeds; Dust K in {64, 256, 1k, 4k} x 3 seeds | 90 | ~30 |
| A (10M tokens) | v1 + int8 added: L4/d256; fp/int8/ternary; backprop 3 seeds + Dust K in {256, 1k} x 3 seeds | 27 | ~45 |
| B+ (analysis) | sigma-Delta sweeps; cosine ladders K in {64, 256, 1k, 4k, 16k} on trained checkpoints; sharpness (absorbs v1 Tier C) | - | ~10 |
| D1 (38M @ 1M tokens) | L8/d512 (upstream default model); fp/ternary; backprop 3 seeds + Dust K in {256, 1k} x 3 seeds | 18 | ~15 |
| D2 (38M @ 10M tokens) | fp/ternary; backprop 3 seeds + Dust K=256 x 3 seeds. With Tier A this gives the 3-point size ladder (2M/7M/38M) at 10M tokens per precision, mirroring Dust's Figure 4 | 12 | ~60-120 |
| D3 (depth) | +2 seeds on the headline cells (7M ternary @ 10M: backprop, K=256, K=1k) | 6 | ~8 |
| D4 (gated extras) | discrete-update variant pilot (T13); weight-space-ZO baseline for RQ5 (T14) | 2-6 | ~15-30 |

Experiment cost is roughly 185-260 T4-equivalent hours, about 90-130 L4-hours, about 150-220 CU at the assumed burn rate [all design estimates; T1 replaces them]. Add v1's infrastructure, tuning and buffer (~25-35 T4-eq) on top. Verdict: fits within 1-2 months of Colab CU if ~1.7 CU/h holds; **D2 is the first thing to cut if the measured rate is worse, D4 second**. v1's 130-hour workshop budget remains valid for the Tier-1 target.

Notes:

- K=64 is not an upstream preset (supported populations: 256, 1024, 4096, 16384 [verified README]). T0/T1 must confirm how to run arbitrary K and the non-default sizes (L2/d128, L4/d256) in the minimal code before promising Tier B's K=64 cells.
- 20M-token runs are dropped from the core path: upstream prepare_data.py builds 1M/10M plus val/test [verified]. 20M is optional later if --tokens accepts arbitrary values [unverified].
- Tier B's L2/d128 serves as the small end of the size axis; Tier A + D2 complete it at 10M tokens.

## 7. Task deltas vs v1

- T0 additions: read full texts of QuZO, QZO, ZeroQAT, HQZO (the positioning in section 2 must survive them); check whether bf16 is hardcoded upstream and whether sizes/populations are configurable; check prepare_data.py argument flexibility; confirm Colab tier exposes L4; extract Appendix C-G into docs/paper_tables.md (v1 requirement).
- T1: benchmark on L4 in bf16; measure the CU burn rate; verify the memory ceiling at K=4k and for the 38M model; single-GPU behavior at K=4k.
- T4 (rewritten): bf16 vs fp32 parity on L4 (loss and cosine within noise). fp16 experiments dropped unless a T4 fallback is needed.
- T6.5 (new): the sigma-Delta sweep of section 4, run after T6 on the same checkpoints; feeds T7.
- T7: tune noise-scale grids anchored at the T6.5 peak; record the chosen sigma/Delta per layer type and precision.
- T12 (v1 journal extension) is replaced by Tier D above.
- T13 (new, gated): discrete-update variant (ternary threshold flips, no latent weights) - pilot at 2M/ternary/1M tokens. Gate G4 below.
- T14 (optional): weight-space ZO baseline for RQ5, one cell.

Gates: G0-G3 unchanged from v1. G1's pass criterion updated: bf16 within noise of fp32 on L4. New G4 after T13: continue the discrete-update line only if the pilot trains at all, or if its failure is cleanly attributable to the section 4 mechanism (a clean negative is paper material given RQ4).

## 8. Statistics upgrades

- 5 seeds on D3 headline cells; 3 seeds everywhere else (v1 rule).
- Paired bootstrap (10,000 resamples) for all Dust-vs-backprop differences; report bootstrap CIs, not only mean and std.
- With 3 seeds, report the honest wide CI rather than a significance claim [design]. The gaps we chase live among loss differences of roughly 0.03-0.5 nats at these scales [paper]; 3 seeds resolve the large ones, D3 exists for the tight ones.

## 9. Paper skeleton v2 (full paper)

Same outline as v1 section 12, plus:

- Figure 5: cosine vs sigma/Delta per layer type, with the fp reference line.
- Figure 6: size ladder at 10M tokens, ternary vs fp, mirroring Dust's Figure 4.
- Table 2: sigma_min and peak sigma/Delta per layer type and precision.
- Related work must engage the QuZO family head-on along two axes: weight-space vs activation-space, and fine-tuning vs from-scratch. Do not bury it.
- Discussion states the STE framing exactly as v1 section 1 and attributes the shared "ZO avoids STE" property honestly to the QuZO line.
- Limitations adds: single-epoch small-token regime inherited from the base protocol; latent-weight updates remain STE-like; compute cost per update is higher than backprop [paper].

Title candidates [design]: "Quantized Transformers Without Backpropagation: Activation-Space Perturbation Under Low-Bit Forwards" / "Backprop-Free Pretraining of Ternary Transformers via Node Perturbation".

## 10. Risk register v2 (deltas only; all v1 section 11 risks still apply)

- Scooping risk RAISED: Dust's own conclusion names non-differentiable and looped architectures as future work [paper]; the QuZO group is adjacent and active. Mitigation: early arXiv after Tier B, narrow claims, the sigma-Delta characterization as defensible core.
- CU burn-rate risk: every L4-hour figure assumes ~1.7 CU/h [unverified]. If the measured rate is 2-3x worse, cut D2 first, then D4, then int8 in Tier A. Re-budget at T1, hard.
- Engineering unknowns in the minimal upstream code [unverified]: arbitrary K (e.g. 64), non-default model sizes, single-GPU K=4k. Discover in T0/T1 before promising those cells.
- Ternary failure at low K is a likely outcome, not a risk to avoid: it is RQ4's mechanism showing up, and publishable if characterized.
- Venue risk for tier 4: scale ceiling (38M params, 10M tokens, one epoch) is the predictable main-track objection; TMLR/journal framing avoids betting the project on it.

## 11. Decisions needed from the owner

1. Venue target: confirm TMLR-first (recommended) vs straight-to-journal vs main-track gamble.
2. Kaggle quota reality: the 45+30 T4 hours [unverified]; after the L4 re-plan, T4 is no longer on the critical path, so this is low priority.
3. Colab tier: confirm L4 access and the CU burn actually observed in the Resources tab.
4. Tier D scope: pre-approve D1; D2 and D4 stay conditional on T1's measured burn rate.
5. Timeline expectation: the full-paper version spans roughly 2-3 calendar months of free-tier compute (Tier B/A in month 1; B+/D1 in month 2; D2 and writing in month 3) [design estimate; owner's free time is the binding constraint].

Cross-references: all unverified items above are collected in docs/OPEN_QUESTIONS.md. This document should be revised, not appended, when T0 and T1 return their reports.



