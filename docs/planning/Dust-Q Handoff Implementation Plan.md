# Dust-Q Handoff: Implementation Plan

Prepared 2026-10-10 for coding and research agents taking over this project. Status: planning only. No code, runs or results exist yet.

## 0. How to use this document

Read all of it before acting. Work through the tasks in section 7 in order and respect the gates in section 10. Rules for every agent:

- Never invent numbers, citations or results. If something is unknown, say so and record it in docs/OPEN\_QUESTIONS.md.
- Tag claims as \[paper\] (stated in the Dust report), \[design\] (our choice, may change) or \[unverified\] (must be checked before relying on it).
- Never tune on the test set. Select checkpoints and hyperparameters on validation only.
- Log every run, including failed or diverged ones. Do not delete bad runs.
- Do not edit the upstream Dust code in place. Keep our changes in our own package or a minimal, documented patch.
- At the end of each task write docs/reports/{task\_id}.md: what was done, commands run, results, deviations, open issues.
- If a requirement is ambiguous or a gate fails, stop and ask the owner instead of improvising.

## 1. Project summary

Owner: an independent student researcher with limited free GPU compute (section 5). Aim: an arXiv preprint and a workshop submission, later extended into a journal paper (Q2 or higher).

Base work: *Dust: Pretraining Transformers Without Backpropagation* (Q Labs, Oct 2026; Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna). Report: https://qlabs.sh/research/dust . Code: https://github.com/qlabs-eng/dust . Appendix: https://qlabs.sh/research/dust/appendix.html .

What Dust does \[paper\]: it is a zeroth-order method that adds Gaussian noise to activations independently at every token (node perturbation). Each token acts as a virtual population member, so one forward pass evaluates thousands of members at once. Each perturbation is rewarded by the loss change it causes. Averaging the reward-weighted noise over K draws estimates the error at a layer's output, and its outer product with the layer's input gives the weight gradient.

Headline results \[paper\], all at small scale (main ladders reach 243M parameters and 20M training tokens; cosine measurements reach 1B tokens):

- Dust ends below tuned backprop at 100k and 1M tokens once K reaches a few hundred to about 1k draws. At 10M and 20M tokens the gap to backprop shrinks as K grows. The 20M extrapolated limit is loosely constrained (95% interval 3.89 to 4.58, against backprop at 4.633).
- Weight-space ES (EGGROLL-Transformer) at a population of 16k still does not reach Dust at 64 draws. The authors extrapolate Dust to be roughly 10^3 to 10^4 times more population-efficient.
- Larger models are often more population-efficient: at 10M tokens the 243M model beats the 2M model at every population above 64 and is about equal at 64.
- The cosine between Dust and backprop gradients rises with K and holds up for most layers from 10M to 1B tokens (measured on backprop-trained checkpoints). Dust also works with Adam at 1M tokens.

Where the paper stops \[paper\]: it does not make Dust compute-efficient enough to replace backprop (the authors say orders of magnitude more efficiency are needed), and it leaves non-differentiable or looped architectures to future work. Our extension targets the second gap with a quantized, non-differentiable forward pass (section 4) \[design\].

Working title: *Backprop-free pretraining of low-bit transformers via activation-space perturbation*.

Research questions:

- RQ1: Can Dust pretrain int8 and ternary (1.58-bit) small transformers, and how does it compare with backprop plus a straight-through estimator (STE) at the same token budget?
- RQ2: How does the gap change with population K, model size (2M vs 7M parameters) and token budget (1M vs 10M)?
- RQ3: Do quantized Dust gradients still follow the paper's cosine-vs-population law, and do Dust and backprop reach minima of different sharpness?

Framing caveat (must appear in the paper): if latent full-precision weights are updated with the Dust weight gradient, the weight-side update is still straight-through. What Dust avoids is STE through activation quantizers and other non-differentiable operations in the credit path. Claim exactly that, nothing stronger.

Negative or mixed results are acceptable if they are carefully characterized (for example, a K below which ternary Dust fails, and why).

## 2. Facts from the Dust report to build on

All items below are \[paper\] unless tagged otherwise.

- Data and protocol: GPT-style transformers on FineWeb, 4096-token BPE tokenizer, batch of 16k tokens (8 sequences of 2048), one epoch, SGD with momentum at constant learning rate. Validation and test sets are 544 sequences each. Report test loss at the best validation checkpoint. Three seeds per cell. Every method is tuned separately at every token budget and population.
- Model sizes: L2/d128 = 2.0M params, L4/d256 = 7.3M, L8/d512 = 38M, L16/d1024 = 243M. The main base model is L8/d512.
- Population is counted in draws K for Dust. One draw is one independent jitter of activations across all tokens of the selected layers.
- Reproduction targets at 10M tokens, test loss for L4/d256 (7.3M): backprop 5.066; Dust K=64: 5.556, K=256: 5.362, K=1k: 5.161, K=4k: 5.095, K=16k: 5.065. Figures for other sizes and the 1M-token budget are in the appendix tables (Appendix C); extract them in T0.
- Dust is not compute-efficient compared with backprop. The authors say so; do not claim otherwise.

## 3. Dust algorithm reference

Do not reimplement Dust; reuse the upstream code. This summary is for understanding and for writing tests.

For a linear layer y\_t = W x\_t, add Gaussian noise to the output at every token independently: y\_t -> y\_t + sigma \* a\_t, a\_t \~ N(0, I). Run the forward pass and compute per-token losses.

```
c_s   = mean_over_draws(loss_s) - loss_s            (centered loss reduction at token s)
r_t   = sum over s >= t of gamma^(s-t) * c_s         (credit with decay gamma)
g_t   = -(1 / (K * sigma)) * sum over draws i of r_t(i) * a_t(i)   (estimated output error)
G_W   = sum over t of g_t x_t^T                      (weight gradient, outer product with the layer input)
```

Details \[paper\]:

- Different layer types are jittered in separate forward passes, each with its own noise scale. The clean forward is cached, and a draw for block l reruns only blocks l onward.
- Attention internals (query, key, value, gate, value embedding) are jittered separately and credited through the estimated error at the attention output: c\_s = -\<g\_s, delta\_o\_s>, with credit from future tokens (gamma close to 1) for keys, values, gates and value embeddings. All other layers use gamma = 0.
- The LM head is jittered on cached logits, re-evaluating only the cross-entropy on a slab of the vocabulary per draw.
- The embedding gradient is a scatter-add of g\_t into the token row. The 2L residual mixing scalars are trained by ordinary weight-space ES.
- Hyperparameters to tune: per-layer-type noise scale, credit decay, and the share of population per layer type. Two tuning modes: short training runs, or maximizing the cosine between the Dust estimate and the backprop gradient on one batch (no training needed).
- Cosine law fitted in the paper: cos(K) = c\_max / sqrt(1 + c / K).

## 4. Our extension: quantized forward pass \[design\]

Precision modes:

- fp: full precision (fp32, or bf16 where the GPU supports it).
- int8: W8A8 fake quantization.
- ternary: W1.58A8 (primary). Optional ablation W1.58A16.

Quantizer definitions (fake quantization, latent fp32 weights kept for updates):

- Ternary weights: gamma = mean(abs(W)) per tensor; Wq = clamp(round(W / (gamma + 1e-5)), -1, 1); effective weight = gamma \* Wq.
- int8 weights: per-tensor absmax scale, round, clamp to \[-128, 127\].
- Activations (int8): per-token absmax over the hidden dimension, scale s = 127 / max(abs(x), 1e-5); xq = clamp(round(x \* s), -128, 127) / s.
- Keep embeddings, LM head, norms and residual mixing scalars in higher precision, as in BitNet-style models.

Training rules:

- Backprop baseline: STE (identity gradient through round and clamp) for weights and activations, latent fp32 weights.
- Dust: same forward pass. Noise is added to linear outputs, so downstream activation quantizers sit inside the forward pass whose loss response is measured and no STE is needed there. The weight gradient G\_W uses the quantized (dequantized-value) input actually fed to the matmul. Apply G\_W to the latent weights with the same rule as the backprop baseline, so the two methods differ only in how the output error g\_t is obtained.
- Both methods use the same optimizer family (SGD with momentum) and separately tuned learning rates.

Optional later ablation: a discrete-update variant (sign or threshold flips of ternary weights without latent weights) to test the no-STE claim more strictly.

## 5. Compute and environment

- Kaggle: owner reports 45 + 30 T4 hours \[unverified: period and whether these are two quotas; confirm with owner\].
- Colab: 200 compute units per month. Recent third-party measurements show T4 about 1.2 to 1.8 CU/hr, L4 about 1.7, A100 40GB about 5.4 to 15 \[unverified, varies with demand\]. Always read the burn rate from the Resources tab and log it.
- Relative speed (estimate): L4 about 2x T4, A100 about 5x or more; small models may not saturate an A100.
- T4 has no bf16. Test fp16 with loss scaling versus fp32 (T4). L4 and A100 support bf16.
- Assignment: Kaggle T4 for debugging, data prep, reproduction and the cheap 1M-token runs. Colab L4 for the long 10M-token runs. A100 only for the 38M journal-extension runs and only if the measured rate is acceptable.
- Session limits: keep each job under 8 hours. Checkpoint every 10 minutes to persistent storage (Drive or Kaggle dataset) and make every run resumable.

## 6. Repository layout and conventions

```
third_party/dust/    upstream repo, pinned commit, unmodified
dustq/               our package: quant.py, dust_adapter.py, cosine.py, sharpness.py, eval.py
configs/             one YAML per run
scripts/             run.py, resume.py, aggregate.py, benchmark.py
results/             runs.csv, per-run logs, checkpoints (not committed)
docs/                REPO_MAP.md, paper_tables.md, OPEN_QUESTIONS.md, reports/
tests/
```

Run naming: {size}*{tokens}*{prec}\_{method}\_K{K}\_s{seed}, for example L4d256\_1M\_ternary\_dust\_K256\_s0. Backprop runs use K0.

Config fields: model (layers, width), data (tokens, seq\_len, batch\_tokens), precision (mode, act\_bits), method (name, population, noise scales, gamma, allocation), optim (name, lr, momentum), seed, dtype, checkpoint\_every\_minutes.

Results table (results/runs.csv) columns: run\_id, task, model, params, tokens, precision, method, K, seed, lr, momentum, noise\_scales, val\_loss\_best, test\_loss\_at\_best\_val, steps, wall\_s, gpu, dtype, cu\_burn\_rate, commit, status, notes.

Determinism: fix seeds for data order, init and perturbations; log library versions; pin dependencies.

## 7. Task list

Each task lists steps, deliverables and acceptance criteria. GPU hours are rough T4-equivalent estimates to be replaced after T1.

### T0. Recon (no GPU)

- Clone the upstream repo, pin the commit, check the license.
- Write docs/REPO\_MAP.md: entry points, model definition, where noise is added, reward and credit computation, population allocation, tuning scripts, data preparation, config system, dtype handling, and any CUDA or flash-attention requirements that break on T4.
- Fetch the appendix (A to G) and put exact tables in docs/paper\_tables.md, especially Appendix C (test losses by token budget and population), D, E and F.
- Run a novelty search on zeroth-order or perturbation-based training combined with quantization; record findings in docs/related\_work.md. Verify, do not assume, what exists.
- Acceptance: REPO\_MAP identifies exactly where quantization hooks go; paper\_tables.md exists; related\_work.md lists at least 10 verified references with links.

### T1. Infrastructure (about 2 GPU hours)

- Config loader, logging to results/runs.csv (optionally W&B), checkpoint and resume, seed control.
- scripts/benchmark.py: tokens per second at K=256 for each GPU type, plus the CU burn rate.
- Acceptance: a run killed mid-way resumes and reaches the same final loss within noise; benchmark table saved in docs/reports/T1.md.

### T2. Data (CPU, then 1 GPU hour)

- Prepare a FineWeb subset with the 4096 BPE tokenizer: 20M training tokens, plus validation and test sets of 544 sequences each (2048 tokens). Use upstream scripts if they exist \[unverified\].
- Acceptance: token counts verified; train, val and test are disjoint; a 1M-token and a 10M-token prefix are defined and fixed.

### T3. Reproduction (about 10 GPU hours)

- Run L4/d256 at 10M tokens: backprop and Dust at K=256, 3 seeds each, then K=1k if hours allow. Use 1M-token runs first for pipeline checks.
- Compare with the targets in section 2 and with Appendix C.
- Acceptance (proposed): means within 0.05 absolute of the paper values and the same ordering across K. If not, check dtype, learning-rate grid, data split and tuning before escalating to the owner.

### T4. Dtype check

- On a 1M-token Dust K=256 run and a single-batch cosine measurement, compare fp32, fp16 with loss scaling, and bf16 (L4 or A100).
- Acceptance: a table stating which dtype is usable per GPU; losses and cosines within noise of fp32.

### T5. Quantization module (about 1 GPU hour for tests)

- Implement the modes in section 4 in dustq/quant.py and integrate through dust\_adapter.py.
- Unit tests: ternary weights take only values in {-1, 0, 1} times gamma; activation quantizer round-trips within one step; STE backprop gradients are finite and nonzero; the Dust forward matches the quantized forward exactly when noise is zero.
- Acceptance: tests pass; a 100-step smoke run per mode trains without NaN.

### T6. Gate G2: quantized cosine (about 5 GPU hours)

- Train backprop+STE checkpoints at 1M tokens (2M and 7M models; fp, int8, ternary).
- On a fixed batch, compute cosine between the Dust estimate and the backprop+STE gradient for K in {64, 256, 1k, 4k, 16k}, per layer type. Fit the paper's cosine law.
- Note that the STE gradient is not the true gradient. Report fp cosine alongside.
- Pass (proposed): cosine rises with K for MLP and attention-output layers, and the fitted c\_max is at least 50 percent of the fp value. Otherwise stop and report.

### T7. Tuning (about 5 GPU hours)

- Tune per-layer-type noise scales with the cosine method on one batch, then a small learning-rate and momentum grid with short training runs.
- Record grid sizes and chosen values per cell; tune each method separately.

### T8. Tier B runs (about 30 GPU hours)

- See section 8. Run the full 1M-token matrix. Aggregate to results/runs.csv and produce a first loss-vs-K plot per precision.

### T9. Tier A runs (about 30 GPU hours)

- See section 8. Use Colab L4 where possible.

### T10. Analysis (about 10 GPU hours)

- Cosine ladders for all precisions; sharpness measurements; loss versus FLOPs; failure-mode notes (divergence, K thresholds).

### T11. Writing support

- Generate figures and tables from results/runs.csv only (scripts must regenerate them end to end). See section 12.

### T12. Journal extension (after workshop feedback; about 40 to 80 GPU hours)

- Add L8/d512 (38M) at 1M tokens for {fp, ternary}, K in {64, 256}, 3 seeds; raise seeds to 5 on the key cells; add ablations (activation bits, noise placement, discrete-update variant).

## 8. Experiment matrix

Tier B (1M tokens): sizes L2/d128 and L4/d256; precisions fp, int8, ternary; methods backprop (3 seeds) and Dust with K in {64, 256, 1k, 4k} (3 seeds each). That is 15 runs per size and precision, 90 runs total.

Tier A (10M tokens): L4/d256; precisions fp and ternary; Dust K in {256, 1k}, 3 seeds each, plus backprop 3 seeds per precision. That is 18 runs.

Tier C (analysis): cosine ladders on backprop-trained checkpoints at 1M and 10M tokens; sharpness on final models.

Tier D (journal extension): see T12.

Estimated budget: T0 to T4 about 15 hours, T5 to T7 about 10, Tier B about 30, Tier A about 30, Tier C about 10, buffer about 35. Total about 130 T4-equivalent hours. Update after T1.

## 9. Metrics and statistics

- Primary: test loss at the best validation checkpoint (as in the paper).
- Secondary: per-layer-type gradient cosine versus backprop(+STE); wall-clock and approximate FLOPs per update; divergence rate.
- Sharpness \[design\]: for each trained model take 20 random directions with filter-wise normalization, evaluate validation loss on 64 sequences at w + rho \* d for rho in {0.01, 0.02, 0.05, 0.1}, report the mean loss increase. For quantized models perturb the latent weights and re-quantize in the forward pass. Optionally estimate the top Hessian eigenvalue by power iteration on the fp 2M model.
- Statistics: report mean and standard deviation over seeds; use a paired bootstrap (10,000 resamples) for Dust minus backprop differences; never report a single seed as a result.

## 10. Decision gates and fallbacks

| Gate | When | Pass | If it fails |
| --- | --- | --- | --- |
| G0 | After T3 | Reproduction within tolerance | Debug; if still off, switch to a reproduction-and-ablation paper |
| G1 | After T4 | A usable dtype per GPU | Use fp32 and halve the planned matrix |
| G2 | After T6 | Cosine rises with K for quantized layers | Try int8 only, or pivot to the sharpness analysis (why Dust sometimes beats backprop) |
| G3 | After Tier B | Any competitive quantized cell, or a clear failure map | Write up as a characterization of failure modes for a workshop |

At each gate write a one-paragraph recommendation for the owner before continuing.

## 11. Risks and unverified items

- Scooping: the Dust authors or others may publish the same extension. Mitigation: post an arXiv preprint early, keep the claim narrow.
- STE framing: reviewers will ask. Address it directly in the method section (section 1).
- fp16 noise on T4 may swamp small loss differences. Decided in T4.
- Free-tier sessions die. Mitigation: checkpoint and resume (T1).
- &#91;unverified\] Repo structure, license, hyperparameters, whether data-prep scripts exist.
- &#91;unverified\] Colab CU rates and Kaggle quota details.
- &#91;unverified\] Prior work on quantized zeroth-order training; T0 must settle this before any novelty claim.
- Dust requires large populations. Do not claim on-device or energy-efficiency benefits without measurements.

## 12. Writing handoff

Paper outline: abstract; introduction; background (Dust, node perturbation, BitNet-style quantization); method (precise definitions from section 4, STE caveat); experiments (setup, results per RQ); analysis (cosine, sharpness, failure modes); limitations (small scale, compute cost, STE-like latent update); related work; reproducibility (code, configs, seeds).

Figures and tables, all generated from results/runs.csv:

- Figure 1: test loss versus K per precision, with backprop reference lines.
- Figure 2: gradient cosine versus K per layer type, fp versus quantized, with fitted laws.
- Figure 3: Dust minus backprop gap versus model size and token budget.
- Figure 4: sharpness comparison.
- Table 1: main results with mean and standard deviation over 3 seeds.

Related work to read and verify before citing: the Dust report; EGGROLL (arXiv:2511.16652); evolution strategies (arXiv:1703.03864); MeZO (Malladi et al., 2023); node perturbation (Werfel et al., 2003); forward gradients with local losses (Ren et al., 2023); BitNet b1.58 \[unverified identifier, look up\]; evolution strategies at scale for LLM fine-tuning (arXiv:2509.24372).

Cite the Dust report as a Q Labs technical report with the URL and acknowledge the authors; follow the repo license.

## 13. Definition of done

Workshop preprint stage is done when: T0 to T10 reports exist; all results are reproducible from configs and seeds; figures and tables regenerate from one script; the paper draft states claims exactly as supported by results; the code and configs are released with a README; and open questions are logged in docs/OPEN\_QUESTIONS.md.
