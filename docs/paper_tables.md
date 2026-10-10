# Dust (Q Labs) — Appendix tables A–G (reference extract)

- **Paper:** "Dust: Pretraining Transformers Without Backpropagation" — Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna, Q Labs Research, October 2026.
- **Primary source:** https://qlabs.sh/research/dust/appendix.html — **Fetched 2026-10-11**
- **Cross-check source (main post, fetched 2026-10-11):** https://qlabs.sh/research/dust/ — used only for items the appendix does not contain (Figure 4 model-size table, Figure 3 Adam numbers, Figure 2 fit annotation). Every such item is explicitly labeled "main post".
- The appendix is a single page; there are no paginated sub-pages. All numbers are copied **exactly as printed**, including trailing zeros (e.g. `1.0000`, `0.00175`) and the printed thousands separators (e.g. `23,000×`, `65,536`). Nothing is rounded, converted, or estimated.
- Where a value or table does not exist in the source, it is marked **[unverified: not readable]** rather than guessed.
- Notation: the paper prints the power-law limit row as `$\infty$ (fit)`; here it is written `∞ (fit)`. In Table 2, `∞` marks "a refit whose floor lies above the loss of Dust".

---

## Appendix A — Architecture

**Source:** Appendix A · Architecture · https://qlabs.sh/research/dust/appendix.html#architecture (fetched 2026-10-11)

No table in this section; verbatim text:

> The model is a decoder-only transformer of the modded-nanoGPT kind: eight pre-norm blocks of width 512 with eight heads, RMS normalization, rotary position embeddings with normalized queries and keys, a squared-ReLU MLP of width 2048, and no biases. Attention is causal with a sliding window of 1024 tokens in three of every four blocks and the full 2048 in the others. Every other block adds a *value embedding* to its values, a second token embedding scaled per head by a learned gate on the block input. The residual stream entering each block is a learned mix of the previous block's output and the token embedding, and these $2L$ mixing scalars are the parameters trained by weight-space ES. The head is untied, with capped logits. Token and value embeddings are stored in bfloat16 and the forward runs in bfloat16. The base model has 37.7M parameters; the model-size study uses two, four and sixteen blocks of width 128, 256 and 1024, with the same head dimension.

Per-size architecture as printed (sizes/codes from main post Figure 4 caption: L2/d128, L4/d256, L8/d512, L16/d1024):

| Model | Layers | Width | Heads (as printed) | Params (as printed) |
| --- | --- | --- | --- | --- |
| Base model (main comparison) | 8 | 512 | 8 | 37.7M |
| L2/d128 (size study) | 2 | 128 | not printed | 2.0M (main post) |
| L4/d256 (size study) | 4 | 256 | not printed | 7.3M (main post) |
| L8/d512 (size study) | 8 | 512 | 8 (same as base model) | 38M (main post) |
| L16/d1024 (size study) | 16 | 1024 | not printed | 243M (main post) |

- Head counts for L2/d128, L4/d256, L16/d1024 are **not printed** anywhere in the paper's web version. The only printed facts are "eight heads" (base, width 512) and "the same head dimension" for the size study. Base head dimension is therefore 512/8 = 64, which *implies* 2, 4 and 16 heads at widths 128, 256, 1024 — but those three counts are **derived, not printed**: [unverified: not readable] as paper values.
- **Missing:** explicit head counts and per-size config (MLP width, window schedule, vocab/embedding sizes) for the size-study models.

---

## Appendix B — Cosine Ladder at the Remaining Checkpoints

**Source:** Appendix B · Cosine Ladder at the Remaining Checkpoints · https://qlabs.sh/research/dust/appendix.html#cosine-ladder-at-the-remaining-checkpoints (fetched 2026-10-11)

No table; one figure only. Verbatim caption:

> **Figure 6:** The cosine ladder of Figure 5 at the checkpoints not shown there. Left, Dust at initialization. Middle and right, EGGROLL at initialization and at 100M tokens, each on its own axis.

Figure file: https://qlabs.sh/research/dust/figures/cospop_appendix.svg

- **Missing:** the plotted cosine values are **[unverified: not readable]** — the SVG renders all text/labels as vector glyph paths with no extractable numbers, and no data table exists. Only the qualitative content (Dust at init; EGGROLL at init and at 100M tokens) is stated.

---

## Appendix C — Test Losses by Token Budget and Population

**Source:** Appendix C · Test Losses by Token Budget and Population · https://qlabs.sh/research/dust/appendix.html#test-losses-by-token-budget-and-population (fetched 2026-10-11)

Verbatim intro: "Table 1 has the test losses behind Figure 2."

**Table 1 (Appendix C)** — *Test loss at the validation selected checkpoint averaged over three seeds, across token budgets and populations. Population is draws rewarded with token losses for Dust and forward passes of the batch for EGGROLL (Section 3.1). Backprop is tuned on the same grid at every budget. The ∞ row is the limit of a power law fit in population.*

| | Population | 100k | 1M | 10M | 20M |
| --- | --- | --- | --- | --- | --- |
| Dust | 64 | 7.240 | 6.290 | 5.397 | 5.265 |
| | 256 | 7.165 | 6.067 | 5.231 | 5.139 |
| | 1k | 7.151 | 5.934 | 5.111 | 4.960 |
| | 4k | 7.164 | 5.911 | 5.072 | 4.912 |
| | 16k | 7.170 | 5.916 | 5.049 | 4.802 |
| | ∞ (fit) | 7.161 | 5.892 | 5.013 | 4.431 |
| EGGROLL | 64 | 7.961 | 7.428 | 7.056 | 6.987 |
| | 256 | 7.605 | 7.200 | 6.906 | 6.811 |
| | 1k | 7.386 | 7.046 | 6.692 | 6.517 |
| | 4k | 7.302 | 6.892 | 6.352 | 6.200 |
| | 16k | 7.263 | 6.708 | 6.033 | 5.865 |
| Backprop | | 7.216 | 5.959 | 4.989 | 4.633 |

Notes (verbatim-supporting context):
- These are **three-seed means** at the validation-selected checkpoint. **No standard deviations are printed anywhere in the appendix** — [unverified: not readable] for per-cell std devs.
- Table 1 covers **one model size only**: the 37.7M-parameter base model (8 layers, width 512 — Appendix A). It is the *main comparison* of Section 3.2.
- 20M-token results (the 20M column): Dust is still falling at 16k draws (4.802); the fitted limit is 4.431 vs backprop 4.633. Main post §3.2 adds: "Its power law fit puts the limit at 4.431 (95% interval 3.89 to 4.58), below backprop's 4.633, but with the ladder still falling the fit is loosely constrained"; the Figure 2 annotation (main post) prints the fit as `1.53 P−0.145 + 4.43`, limit `4.43`.

### Cross-check requested against the main post (Figure 4, 10M tokens, L4/d256 = 7.3M)

**The model-size table is NOT in the appendix.** The four sizes (2.0M, 7.3M, 38M, 243M) × populations × backprop-per-size appear only in **Figure 4 of the main post** (https://qlabs.sh/research/dust/#overparameterization, fetched 2026-10-11), at a fixed 10M tokens. Verbatim transcription of that figure's table (bold = best size in row, as printed; the "Across sizes" sparkline column contains no numbers and is omitted):

**Figure 4 (main post — not appendix content)** — *Four model sizes at a fixed 10M tokens, 2M to 243M parameters (L2/d128, L4/d256, L8/d512, L16/d1024), test loss against population. The table has the same cells and backprop per size, with the best size at each population in bold.*

| Population | 2.0M (L2/d128) | 7.3M (L4/d256) | 38M (L8/d512) | 243M (L16/d1024) |
| --- | --- | --- | --- | --- |
| 64 | 5.705 | **5.556** | 5.558 | 5.719 |
| 256 | 5.486 | 5.362 | **5.358** | 5.419 |
| 1k | 5.265 | 5.161 | **5.158** | 5.214 |
| 4k | 5.189 | 5.095 | **5.053** | 5.124 |
| 16k | 5.171 | 5.065 | **5.036** | 5.086 |
| Backprop | 5.180 | 5.066 | **5.015** | 5.048 |

**Cross-check verdict: PASS, no inconsistency.** The supplied expected values for L4/d256 at 10M tokens — backprop 5.066; Dust 64: 5.556, 256: 5.362, 1k: 5.161, 4k: 5.095, 16k: 5.065 — match the figure's 7.3M column **exactly**.

**Do not confuse the two 10M tables:** Appendix C Table 1 (37.7M main-comparison model) has *different* 10M values (Dust 64: 5.397 … Backprop: 4.989) than Figure 4's 38M column (5.558 … 5.015). This is not a contradiction: Appendix F states the model-size study "predates the protocol of Section 3.1" and used the earlier training settings of Table 6, "which explains the difference from the main results."

- **Missing:** model-size test losses at any token budget other than 10M ([unverified: not readable] — not in the paper); per-seed values, standard deviations, and seed counts for the model-size study ([unverified: not readable] — not stated; the study predates the three-seed protocol); EGGROLL rows in the model-size study (not run/not printed).

---

## Appendix D — EGGROLL's Extrapolated Population

**Source:** Appendix D · EGGROLL's Extrapolated Population · https://qlabs.sh/research/dust/appendix.html#eggrolls-extrapolated-population (fetched 2026-10-11)

Verbatim text:

> We extrapolate EGGROLL's ladder at 1M tokens and more to the population at which it would match Dust at 64 draws (Section 3.2). Table 2 gives that population as a multiple of 64 under three fits of EGGROLL's cells in Table 1. The power law is the fit of Figure 2, $L(P) = A P^{-\alpha} + E$, which we also refit with each population left out in turn. The straight lines in log population are least-squares fits to the top three populations (1k, 4k and 16k) and the top two (4k and 16k). The three fits put the multiple between about 3,000 and 23,000.

**Table 2 (Appendix D)** — *The population EGGROLL would need to match the test loss of Dust at 64 draws, as a multiple of 64, from fits of its five cells in Table 1. ∞ marks a refit whose floor lies above the loss of Dust.*

| Fit | 1M | 10M | 20M |
| --- | --- | --- | --- |
| Power law, all five populations | 23,000× | 16,000× | 8,100× |
| Power law, one population left out | 11,000× to ∞ | 7,200× to 54,000× | 4,700× to 17,000× |
| Straight line in log population, top three | 8,200× | 3,700× | 3,300× |
| Straight line in log population, top two | 6,000× | 4,100× | 3,100× |

- **Missing:** the underlying power-law coefficients (A, α, E) of the EGGROLL fits are not printed. [unverified: not readable]

---

## Appendix E — Population Allocation

**Source:** Appendix E · Population Allocation · https://qlabs.sh/research/dust/appendix.html#population-allocation (fetched 2026-10-11)

Verbatim text (first paragraph):

> $K$ counts the draws rewarded with token losses. Table 3 lists everything an update jitters, with its draws at $K = 16{,}384$ and its noise scale, and Table 4 splits the three per-block streams over the blocks. Populations from 256 up scale every count by $K/16{,}384$. At 64 draws the recipe at 1M and 10M is $(8,7,4,3,1,1,1,1)$ writer draws, $(8,7,4,3,2,1,1,1)$ MLP draws, one attention-output draw per block, three embedding draws and 1, 5, 2, 3 and 2 draws for $q$, $v$, the gate, $k$ and the value embedding. The nominal stream ratio before rounding is $(\text{writers}+\text{MLP}):\text{attention outputs}:\text{embedding} = 7:0.75:0.25$ (at 100k tokens $6:1.5:0.5$), with a tilt toward the first blocks that comes from a coordinate descent over the allocation at 2k draws.

**Table 3 (Appendix E)** — *What one update jitters, at $K = 16{,}384$. The first four rows are rewarded with token losses and add up to $K$ under SGD and to $11K/8$ in the Adam ladder. The SGD column is the allocation at 1M tokens, and Table 4 has the other budgets. The attention internals are credited through the attention output, and gates and value embeddings exist only on every other block. Each head draw jitters one slab of 256 logits. The attention outputs of the deeper half use 0.4 because their loss changes are weak and poorly resolved at 0.2. An update also runs one clean forward per GPU.*

(As printed, the "Draws" header spans the SGD and Adam columns; the noise-scale column has its own header.)

| Site | One draw reruns | Draws — SGD | Draws — Adam | Noise scale |
| --- | --- | --- | --- | --- |
| Writer pair | its block onward | 7,040 | 4,096 | 0.2 |
| MLP hidden layer | its block onward | 7,040 | 4,096 | 0.2 |
| Attention output | its block onward | 1,664 | 6,144 | 0.2, deeper half 0.4 |
| Token embedding | the whole network | 640 | 8,192 | 0.2, Adam 0.4 |
| Query, per block | its block's attention | 128 | 128 | 0.05 |
| Value, per block | its block's attention | 1,280 | 1,280 | 0.05 |
| Gate, per block | its block's attention | 384 | 384 | 0.05 |
| Key, per block | its block's attention | 768 | 768 | 0.05 |
| Value embedding, per block | its block's attention | 512 | 512 | 0.05 |
| Head | one slab's cross-entropy | 65,536 | 49,152 | 0.05 |
| Residual scalars | the whole network, twice | 8 | 8 | 0.03 |

(Consistency checks pass: SGD first four rows sum to 16,384 = K; Adam first four rows sum to 22,528 = 11K/8.)

**Table 4 (Appendix E)** — *Draws per block at $K = 16{,}384$ for the writer pair (the MLP hidden layer gets the same, except block 2 at 100k, which gets 1,664) and for the attention output. The 16k cells at 10M and 20M have allocations of their own; every other cell scales its budget's column by $K/16{,}384$. The token embedding gets 640 draws, and 1,152 at 100k. The 10M and 20M allocations at 16k take the same time per update as the one they replace.*

(As printed, a spanner header groups columns 2–6 under "Writer pair, MLP hidden layer" and columns 7–9 under "Attention output", over the column labels below.)

| Block | 100k | 1M, 10M | 20M | 10M at 16k | 20M at 16k | 100k | all others | 10M at 16k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | 2432 | 2816 | 2176 | 2304 | 896 | 384 | 768 |
| 2 | 1792 | 2048 | 2432 | 1792 | 2048 | 768 | 384 | 640 |
| 3 | 896 | 1024 | 896 | 896 | 768 | 512 | 256 | 512 |
| 4 | 640 | 640 | 256 | 512 | 256 | 384 | 128 | 256 |
| 5 | 256 | 384 | 256 | 384 | 256 | 128 | 128 | 256 |
| 6 | 256 | 256 | 128 | 256 | 128 | 128 | 128 | 256 |
| 7 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 256 |
| 8 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 256 |
| Total | 6144 | 7040 | 7040 | 6272 | 6016 | 3072 | 1664 | 3200 |

(Columns 2–6 = writer pair / MLP hidden layer; columns 7–9 = attention output. Row sums match the printed Total row.)

Verbatim text (closing paragraph):

> Draws are evaluated $n$ at a time on one GPU, with $n = 1, 2, 8, 8, 16$ at $K = 64, 256, 1{,}024, 4{,}096, 16{,}384$, and split evenly across GPUs. Direct loss estimates and queries use $\gamma=0$. Keys, values, gates and value embeddings use $\gamma=0.99$ in the main comparison's 10M/16k and 20M/16k runs, and $\gamma=0.98$ in all other runs. At 20M and 16k draws, $k$ and $v$ get 1,536 and 2,560 draws per block, so that cell has 14,336 draws rewarded with losses at the compute of the other 16k cells. For the other model sizes the tilt is sampled at the same relative depth and the attention-internal counts are scaled so that their totals match the 8-layer model.

- **Missing:** no per-model-size draw allocation table is printed (only the scaling rule in the last sentence); the 20M-at-16k attention-output and head draw counts specific to that cell beyond what Tables 3–4 state. [unverified: not readable] for anything per-size beyond the quoted rule.

---

## Appendix F — Tuning

**Source:** Appendix F · Tuning · https://qlabs.sh/research/dust/appendix.html#tuning (fetched 2026-10-11)

Verbatim text (tuning protocol):

> Every cell of the main comparison, a token budget and a population for Dust and a token budget for backprop, is tuned the same way. Both methods tune momentum and learning rate only, starting from the grid of Table 5. Every other constant is fixed (Section 3.1, Appendix E), the learning rates of Table 6 included. Seed 42 runs the grid; the best point and every point within 0.02 of it get seeds 43 and 44, as does any point whose single seed beats the current three-seed best. A best point on the edge of the grid has the grid extended one step along its line until the best point is interior, and the two rate half-steps around it are then run under the same rules. A run whose final validation loss exceeds its best by more than 0.25, or that ever exceeds the uniform-prediction loss $\ln V$, or produces a NaN, is rejected together with its configuration. The selected configuration is the lowest mean validation loss over the three seeds, and we report its test loss at each seed's best-validation checkpoint.

**Table 5 (Appendix F)** — *The 12-point grid each cell starts from. The 64-draw cells at 10M and 20M need heavy momentum and small rates.*

| Cells | Momentum | Learning rate |
| --- | --- | --- |
| 100k tokens | 0.2, 0.3, 0.5 | 0.3, 0.4, 0.5, 0.6 |
| 1M, 10M and 20M tokens | 0.9, 0.95, 0.98 | 0.1, 0.15, 0.2, 0.25 |
| 10M tokens at 64 draws | 0.95, 0.98, 0.99 | 0.0125, 0.025, 0.05, 0.1 |
| 20M tokens at 64 draws | 0.95, 0.98, 0.99 | 0.00625, 0.0125, 0.025, 0.05 |

**Table 6 (Appendix F)** — *Learning rates under SGD. Base is the tuned rate of the cell. The model-size study of Section 4.1 predates the protocol of Section 3.1, and its learning rate and momentum are selected per size and population.*

(As printed, a spanner header groups columns 2–3 under "Main comparison" and columns 4–5 under "Model-size study", over the Dust/Backprop labels below.)

| Parameters | Dust | Backprop | Dust | Backprop |
| --- | --- | --- | --- | --- |
| Key projections | base | base | 4× base | 4× base |
| Value projections | base | base | 8× base | 8× base |
| Token embedding | 1000 | 1000 | 0.3 | base |
| Value embeddings | 0.3 | base | 0.3 | base |
| Residual scalars | 0.03 | base | 0.03 | base |
| Everything else | base | base | base | base |

(Columns 2–3 = Main comparison; columns 4–5 = Model-size study.)

Verbatim text (model-size study settings):

> The overparameterization experiments were run before we improved the token-embedding learning rate. They use the earlier training settings in Table 6, which explains the difference from the main results.
>
> Dust's own constants, the noise scales, the credit decay and the draw allocation, are chosen with the cosine diagnostic of Section 2.3, confirmed in training, and held fixed across cells. At the 16k cells of 10M and 20M tokens they are tuned in training together with momentum and rate at fixed compute: each lever, the allocation across layer families, the per-block tilt, the noise scales, the credit decay and the head draws, is run one at a time from the (momentum, rate) winner, seeded in stages on wins paired by seed, and the winning levers are stacked; backprop receives the same wider (momentum, rate) grid at three seeds. The selected constants are those of Appendix E. Not every lever ran at 20M within the campaign, and the other cells keep the constants from the cosine tuning, so at those two budgets the top cell of the ladder has constants of its own, which the 10M fit is read with.
>
> Under Adam, three learning-rate groups, matrices and the head, token and value embeddings, and residual mixing scalars, are tuned for Dust and backprop. Table 7 has the selected settings at 1M tokens.

**Table 7 (Appendix F)** — *Selected settings under Adam at 1M tokens. Dust's matrix rate is selected per population. All three methods use $\beta_2 = 0.95$, $\epsilon = 10^{-8}$ and no weight decay, and EGGROLL uses z-scored differences.*

| | Dust | Backprop | EGGROLL |
| --- | --- | --- | --- |
| Matrices and head, at 64 draws | 0.00175 | 0.002 | 0.003 |
| Matrices and head, from 256 to 4k | 0.0015 | 0.002 | 0.003 |
| Matrices and head, from 16k up | 0.002 | 0.002 | 0.003 |
| Token and value embeddings | 300× matrices | 0.3 | 0.003 |
| Residual scalars | 100× matrices | 0.2 | 0.003 |
| $\beta_1$ | 0.7 | 0.8 | 0.5 |

Verbatim text (closing paragraph):

> The model-size study of Section 4.1 changes two hyperparameters from Section 3 (Table 6). The token embedding stays at its default learning rate, 0.3 for Dust and the base rate for backprop, because at 1000 backprop shows no clear separation between model sizes, which makes overparameterization hard to study. We also keep the key and value learning rates of the earlier protocol and tune them for both Dust and backprop, so the comparison is fair to both.

### Direct answers to the specific questions asked

- **L4/d256 (7.3M-parameter) configuration:** **[unverified: not readable]** — the appendix prints no per-model-size configuration. What is printed for the model-size study: its learning rate and momentum are "selected per size and population" (Table 6 caption — the selected values themselves are **not listed**), and its per-parameter learning rates are Table 6's "Model-size study" columns (key projections 4× base, value projections 8× base, token embedding 0.3 for Dust / base for backprop, value embeddings 0.3 / base, residual scalars 0.03 / base, everything else base).
- **Number of attention heads (L4/d256):** not printed. Appendix A gives "same head dimension" and the base model's "eight heads" at width 512 (head dim 64), which *implies* **4 heads** at width 256 — derived, not printed. Treat 4 as **[unverified: not readable]** as a paper value.
- **Draw allocation / recipe for L4/d256:** not printed. The only printed rule is Appendix E's: "For the other model sizes the tilt is sampled at the same relative depth and the attention-internal counts are scaled so that their totals match the 8-layer model." The full recipe in Tables 3–4 is for the 8-layer model.
- **Noise scales, and whether they differ by model size:** the printed noise scales (Table 3) are per layer type and per optimizer — 0.2 for writer pair, MLP hidden layer, attention output (deeper half 0.4) and token embedding (0.4 under Adam); 0.05 for query, value, gate, key, value embedding and head; 0.03 for residual scalars. **The appendix never states that noise scales differ across model sizes.** It states they are "held fixed across cells" (main comparison) and re-tuned in training only at the 16k cells of 10M and 20M tokens. Any per-size noise-scale claim is **[unverified: not readable]**.
- **L8/d512 1M-token SGD optimizer settings (learning rate, momentum):** **[unverified: not readable]** — the selected (momentum, learning rate) winners are not printed for any SGD cell. Printed for 1M cells: the starting grid (Table 5: momentum 0.9, 0.95, 0.98; learning rate 0.1, 0.15, 0.2, 0.25) and the per-parameter rates (Table 6). Under **Adam** at 1M tokens the selected settings *are* printed (Table 7 above).
- **20M-token notes in the appendix:** the 20M-at-64-draws grid row (Table 5); "Not every lever ran at 20M within the campaign" (so the 20M/16k cell's constants are partial); the 20M columns of Tables 1, 2, 4; the 20M/16k γ=0.99 and the 1,536/2,560 k/v draws (Appendix E).
- **Adam results:** the appendix contains only settings (Table 7) and the Adam draw allocation (Table 3, Adam column) — no Adam loss table. Main post Figure 3 (fetched 2026-10-11, labeled main post) reports at 1M tokens: Dust's fitted limit **5.248** (95% bootstrap interval **5.17 to 5.30**), backprop **5.361**; means over five seeds for Dust and backprop, three for EGGROLL; "the Adam runs of Dust give the token embedding more draws, about 1.4× the listed population in total"; EGGROLL's tuned Adam ladder lands within 0.01 of its SGD ladder at every population (§3.3).

- **Missing:** selected SGD (momentum, rate) per cell; any per-size tuning values; EGGROLL's own grid values (step size, momentum, noise scale, fitness shaping — mentioned on the main post but never tabulated). All **[unverified: not readable]**.

---

## Appendix G — Cosine-fit parameters

**Source:** Appendix G · Cosine-fit parameters · https://qlabs.sh/research/dust/appendix.html#cosine-fit-parameters (fetched 2026-10-11)

Verbatim text:

> Table 8 reports fits of $C(K) = c_{\max}/\sqrt{1+c/K}$ to measured cosine similarities at the final 100M token checkpoint. Each fit uses seven population budgets from 64 to 131072. At each budget, cosine similarity is computed separately for each weight matrix and then averaged across matrices of the same layer type. The fitted parameter $c$ uses the same population-budget units as $K$, rather than the individual draw count of a particular layer. RMSE is computed across the seven measured points used for fitting.

**Table 8 (Appendix G)** — *Fitted parameters and errors for cosine similarity as a function of population size. Fits describe layer-type means.* (The row in italics is the printed section header inside the table.)

| Layer type | $c_{\max}$ | $c$ | RMSE |
| --- | --- | --- | --- |
| *100M-token checkpoint* | | | |
| MLP out | 0.8779 | 31623 | 0.0081 |
| MLP in | 0.8577 | 48417 | 0.0065 |
| Attn out | 0.8729 | 21627 | 0.0093 |
| Attn Q | 1.0000 | 76736 | 0.0356 |
| Attn K | 1.0000 | 616595 | 0.0191 |
| Attn V | 0.8776 | 173780 | 0.0269 |
| Value gate | 0.6657 | 47863 | 0.0367 |
| Value emb | 1.0000 | 495450 | 0.0247 |
| Token emb | 0.7318 | 19498 | 0.0083 |
| LM head | 1.0000 | 759 | 0.0032 |

Consistency check (main post §4.2, labeled main post): "the fits to layer type mean cosine similarities have RMSEs of 0.0032–0.0367" — matches the RMSE column's min/max exactly. Per-matrix (not layer-type-mean) fits are not printed.

- **Missing:** fit parameters for the other checkpoints — 10M tokens, 1B tokens, initialization, and EGGROLL — are shown only as curves (main post Figure 5; Appendix B Figure 6) and are **not tabulated** anywhere: [unverified: not readable]. Only the 100M-token checkpoint is tabulated.

---

## Gaps and open items

1. **Model-size study table is not in the appendix.** Test losses by model size (2.0M, 7.3M, 38M, 243M) exist only in main-post Figure 4, only at 10M tokens, with no standard deviations and no stated seed count (the study predates the three-seed protocol). All other model-size cells: [unverified: not readable].
2. **No per-cell standard deviations anywhere** in the appendix; Table 1 reports three-seed means only. [unverified: not readable]
3. **L4/d256 head count not printed** — only "same head dimension" (implying 4 heads, derived). [unverified: not readable]
4. **L4/d256 draw allocation and noise scales not printed** — only the 8-layer recipe (Tables 3–4) plus the per-size scaling rule. Any per-size noise-scale claim is unsupported by the paper text; the appendix does not say noise scales differ across model sizes.
5. **Selected SGD (momentum, learning rate) per cell not printed** — including L8/d512 at 1M tokens. Only starting grids (Table 5) and per-parameter multipliers (Table 6) are printed; Adam 1M winners are printed (Table 7). [unverified: not readable]
6. **Appendix G tabulates only the 100M-token checkpoint.** Fits for 10M, 1B, initialization and EGGROLL are figure-only. [unverified: not readable]
7. **Appendix B has no readable numbers** — the SVG's text is vector paths; no data table exists. [unverified: not readable]
8. **Adam losses are not tabulated in the appendix** — only settings; losses are in main-post Figure 3 (limit 5.248, backprop 5.361, interval 5.17–5.30).
9. **EGGROLL's tuning grid** (step size, momentum, noise scale, fitness shaping) is mentioned on the main post but never tabulated. [unverified: not readable]
10. **Cross-check outcome (recorded for the parent task):** the expected L4/d256 values (backprop 5.066; Dust 5.556 / 5.362 / 5.161 / 5.095 / 5.065 at 64/256/1k/4k/16k) match main-post Figure 4 exactly — no inconsistency. The different 10M values in Appendix C Table 1 belong to the 37.7M main-comparison model under the newer protocol (Appendix F explains the difference).
11. Possible follow-up source for the missing per-cell numbers: the paper's code repository, https://github.com/qlabs-eng/dust (linked from the main post; not fetched in this task).
