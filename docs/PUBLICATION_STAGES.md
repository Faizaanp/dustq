# Publication stages — "if I stop now, what is publishable, and where?"

Living document for the owner. Read this before deciding to stop at any point.

**Update rule:** whenever a gate (G0–G4) is decided or a phase completes, update the "Current stage" block in the same commit as the gate recommendation (Plan v3.1 section 8). Venue odds below are judgment calls tagged [design judgment]: this map says what the *evidence* supports, never what will be accepted.

## Current stage (updated 2026-10-11)

**Stage 1 reached — Phase 0 complete** (see `docs/reports/T0.md`): upstream pinned and mapped, hypotheses pre-registered, all CPU acceptance tests green (DUSTQ bitwise-equal to DUST), appendix targets pinned, novelty niche confirmed open as of 2026-10-11 (31 verified references; top overlap risk: FOGZO). Nothing is publishable yet. The first real publishability checkpoint is **Stage 3** (after T6/G2a); the cheapest *meaningful* stop is **Stage 4** (the ~45 L4-h minimum paper, Plan v3.1 section 6.3).

## Stage map

| Stage | You have stopped after | Evidence in hand | If you stop here, it supports | Venue reality |
| --- | --- | --- | --- | --- |
| 0. Plan | 2026-10-10 | verified plan; no runs, no data | nothing publishable — a plan is not a paper | — |
| 1. Recon | T0, T0-lit, T0-hyp + CPU acceptance | niche confirmed or killed by the deep literature pass; hypotheses pre-registered; upstream pinned and mapped | nothing publishable standalone; near-zero cost to have learned it | — |
| 2. Reproduction | T1–T4 (gates G0, G1) | Tier R: internal Dust + backprop reproduction on the minimal code; dtype settled; measured s/step and CU burn; re-cut budget | a reproduction *appendix* for the paper, not a standalone paper; an optional reproducibility-track writeup (workshop poster / ReScience-style) [design judgment] | enabler only |
| 3. Probes | T5–T6 (gates G2a/G2b) | quantized forward implemented; STE/SSTE-bias probe, σ–Δ sweep, directional descent, cosine ladders on real 1M checkpoints | probe-only short paper: "is STE bias real at A4/A2, and is Dust's advantage just smoothing?" — no training claims needed | workshop: yes; TMLR: possible if directional descent is decisive [design judgment]. This is exactly Plan v3.1's G2-fallback outcome |
| 4. Minimum paper | T7–T8 (gate G3), ≈45 L4-h | + B-S tier incl. BP-SSTE; primary interaction with CI; K axis at 1M tokens | H1–H3 tested. **arXiv preprint goes up here at the latest** (post-G3, Plan v3.1 section 11) | TMLR submission scoped as "a 1M-token study"; if G3 fails to resolve the contrast: workshop characterization paper |
| 5. Core complete | T9–T10, ≈80 L4-h | + B-M, Tier A (10M tokens), D2 seeds on the primary cell and its fp counterpart | full TMLR submission with size/token/K scaling; Q2-journal plausible [design judgment] | TMLR: solid; Neurocomputing-class journal: plausible |
| 6. Extension | T11 (D1–D5) | + 38M at 10M tokens; discrete-update pilot; BP-LSQ / ZO-W / checkpoint-continuation extras | the journal version | Neural Networks / Pattern Recognition / Knowledge-Based Systems; IEEE TNNLS a stretch (Plan v3.1 section 1); conference main track only if a result exceeds the plan's claims — not planned |

## What each stop-write contains

- **Stage 3 write:** Fig 1 (STE/SSTE bias vs bits), Fig 2 (σ–Δ sweep + flip rate), Fig 4 (directional descent), cosine ladders. No training claims. The framing caveat in the intro.
- **Stage 4 write:** the above + Fig 5 (loss vs K per precision with BP/BP-STE/BP-SSTE lines), Fig 6 (primary interaction — the headline), Table 1. H2 is claimed only if the paired-bootstrap CI excludes 0 (Plan v3.1 section 9).
- **Stage 5–6 writes:** the above + Fig 7 (size/token axes), Table 2 (σ\*/Δ), D-tier tables. Tier A and B-M become revision-stage additions if you stopped at Stage 4 and resumed later.

All figures/tables regenerate from `results/runs.csv` (Plan v3.1 T12); no number may appear that the repo cannot regenerate.

## Hard floors (violating any kills publishability at every stage)

- Never a single-seed result; no test-set tuning (validation-only selection); every reported number regenerates from configs + seeds; no run deleted (divergence is a reported outcome); claims stated exactly, including the STE framing caveat (Plan v3.1 section 3.2); code and configs released by submission time; deviations from the plan listed in the paper's appendix.

## Venue cheat sheet [unverified: check every deadline when the stage completes]

- **Workshops:** NeurIPS / ICLR / ICML workshops on efficient training, forward-only/zeroth-order methods, or quantization. Calls cluster around mid-year and autumn; check current cycles when a stage lands.
- **TMLR:** rolling submission; judged on evidence, not scale or novelty claims — the natural home for Stages 4–5.
- **Journals (Stages 5–6):** Neural Networks, Neurocomputing, Pattern Recognition, Knowledge-Based Systems (Q2+); IEEE TNNLS is the stretch.
- **arXiv:** post after G3 (Stage 4) to claim priority — the main scooping mitigation (Plan v3.1 section 11).
