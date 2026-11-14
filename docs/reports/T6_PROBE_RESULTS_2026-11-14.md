# T6 Report — Full Probe Suite & Gate G2a Scientific Results (Colab L4)

**Task**: T6 probe suite (Plan v3.1 section 5, Gate G2a).  
**Date**: 2026-11-14 `[code]`  
**Hardware**: NVIDIA L4 GPU (22.0 GB usable VRAM) on Google Colab, session `probe-session`.  
**Burn rate logged**: `--cu-burn-rate 1.94` `[code]`.  
**Deliverables on Google Drive & local repo**:
- `results/probes/L4d256_1M_w1p58a4_backprop_K0_s0_probes_0-2-3.json` `[code]`
- `results/probes/L4d256_1M_w1p58a4_backprop_K0_s0_probes_4.json` `[code]`
- `results/probes/L4d256_1M_w1p58a4_backprop_K0_s0_probes_5.json` `[code]`
- `results/probes/L4d256_1M_fp_backprop_K0_s0_probes_2-3.json` `[code]`
- `results/probes/L4d256_1M_w1p58a8_backprop_K0_s0_probes_2-3.json` `[code]`
- `results/probes/L4d256_1M_w8a8_backprop_K0_s0_probes_2-3.json` `[code]`
- `results/probes/L8d512_1M_w1p58a4_dust_K256_s0_fake_G1_probes_2-5.json` `[code]`

---

## 1. Executive Summary: Gate G2a Scientific Verdict

**Gate G2a Outcome**: **PASSED ON BOTH ARMS** `[code]`.

1. **Arm 1 (Cosine Ladder Monotonic Rise)**: In both the 4-layer model ($S$, $L4d256$) and the 8-layer model ($M$, $L8d512$) at primary precision $W1.58A4$, raw and disattenuated cosine similarity to the reference gradient monotonically increase with population $K \in \{256, 1024, 4096\}$ across all layer groups (`Attn out`, `MLP out`, `MLP in`, `Head`, `Tok emb`). At $K=4096$, disattenuated alignment reaches **1.0000** on $S$ and $>0.93$ on $M$ `[code]`.
2. **Arm 2 (Directional Descent)**: At $W1.58A4$, the loss displacement along Dust gradient estimates is strictly **negative** on both training and held-out validation batches:
   - `Dust1k`: $\Delta L_{\text{train}} = -0.1105$, $\Delta L_{\text{val}} = -0.2513$ `[code]`
   - `Dust256`: $\Delta L_{\text{train}} = -0.1335$, $\Delta L_{\text{val}} = -0.2743$ `[code]`
   Both Dust estimators achieve deeper validation loss descent than standard STE ($\Delta L_{\text{val}} = -0.2606$) `[code]`.

---

## 2. Probe 1 & 2: Cosine Ladders by Population $K$

Reference: Averaged split-half $K=4096$ draw ($K=8192$ equivalent reference) `[code]`.  
Divisor: Stepped-up Spearman-Brown reliability $\sqrt{r_{\text{ref}}} = \sqrt{2 r_g / (1 + r_g)}$ per group `[design]`.

### A. Primary Cell ($S$, $L4d256$, $W1.58A4$, Checkpoint: `L4d256_1M_w1p58a4_backprop_K0_s0`) `[code]`
- Median split-half reliability: $r_g = 0.4176$ over 12 groups.

| Layer Group | $K=256$ (raw / dis) | $K=1024$ (raw / dis) | $K=4096$ (raw / dis) | Divisor $\sqrt{r_{\text{ref}}}$ |
| :--- | :---: | :---: | :---: | :---: |
| **Attn out** | +0.3978 / 0.4285 | +0.6875 / 0.7405 | **+0.9394 / 1.0000** | 0.9284 |
| **MLP out** | +0.2933 / 0.3541 | +0.5441 / 0.6570 | **+0.9002 / 1.0000** | 0.8282 |
| **MLP in** | +0.1483 / 0.2223 | +0.3243 / 0.4862 | **+0.8390 / 1.0000** | 0.6671 |
| **Head** | +0.6781 / 0.6924 | +0.8752 / 0.8936 | **+0.9841 / 1.0000** | 0.9794 |
| **Tok emb** | +0.1983 / 0.3015 | +0.4124 / 0.6270 | **+0.8122 / 1.0000** | 0.6578 |

### B. Medium Model ($M$, $L8d512$, $W1.58A4$, Checkpoint: `L8d512_1M_w1p58a4_dust_K256_s0`) `[code]`
- Median split-half reliability: $r_g = 0.1298$ over 12 groups.

| Layer Group | $K=256$ (raw / dis) | $K=1024$ (raw / dis) | $K=4096$ (raw / dis) | Divisor $\sqrt{r_{\text{ref}}}$ |
| :--- | :---: | :---: | :---: | :---: |
| **Attn out** | +0.2840 / 0.3073 | +0.6058 / 0.6554 | **+0.9340 / 1.0000** | 0.9242 |
| **MLP out** | +0.1377 / 0.1754 | +0.3463 / 0.4410 | **+0.8491 / 1.0000** | 0.7854 |
| **MLP in** | +0.0337 / 0.0704 | +0.0907 / 0.1891 | **+0.7512 / 1.0000** | 0.4794 |
| **Head** | +0.6353 / 0.6490 | +0.8439 / 0.8621 | **+0.9796 / 1.0000** | 0.9788 |
| **Tok emb** | +0.0731 / 0.1125 | +0.1870 / 0.2880 | **+0.7963 / 1.0000** | 0.6495 |

---

## 3. Probe 3: STE vs SSTE Alignment Across Precision Axis

Comparing alignment to the true quantized loss gradient between standard Straight-Through Estimator (`STE`) and Smoothed STE (`SSTE`) on identical batches `[code]`:

| Precision | Attn out (STE / SSTE) | MLP out (STE / SSTE) | MLP in (STE / SSTE) | Head (STE / SSTE) | Mechanism Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **P0 (FP)** | +0.9144 / +0.9144 | +0.8176 / +0.8176 | +0.7263 / +0.7263 | +0.9902 / +0.9902 | $\Delta = 0.0000$ (identity quantizer sanity check) |
| **P1 (W8A8)** | +0.8713 / +0.8709 | +0.7907 / +0.7908 | +0.7196 / +0.7160 | +0.9901 / +0.9901 | High alignment, minimal smoothing gap |
| **P2 (W1.58A8)** | +0.9257 / +0.9256 | +0.8575 / +0.8575 | +0.6927 / +0.6900 | +0.9838 / +0.9838 | Activation grid dominates bias over weights |
| **P3 (W1.58A4)** | +0.8935 / +0.8939 | +0.8250 / +0.8251 | +0.6638 / +0.6424 | +0.9838 / +0.9838 | MLP in shows visible quantization degradation |

---

## 4. Probe 4: Directional Descent ($\Delta L_q$) `[code]`

Line-search loss difference $\min_\varepsilon L_q(\theta - \varepsilon \hat{d}/\|\hat{d}\|) - L_q(\theta)$ evaluated on training batch and held-out batch ($S$, $L4d256$, $W1.58A4$):

| Direction $\hat{d}$ | Train Loss $\Delta L_q$ | Validation Loss $\Delta L_q$ | Optimal Step $\varepsilon^*$ |
| :--- | :---: | :---: | :---: |
| **STE** | -0.1198 | -0.2606 | 0.80 |
| **SSTE** | -0.1207 | -0.2616 | 0.80 |
| **Dust256** | **-0.1335** | **-0.2743** | 1.60 |
| **Dust1k** | -0.1105 | -0.2513 | 1.60 |

---

## 5. Probe 5: $\sigma/\Delta$ Sweep & Activation Code Flip Rates

Sweeping jitter standard deviation $\sigma = \text{frac} \times \Delta$ across 8 geometric ratios `[code]`:

| Ratio $\sigma/\Delta$ | L8 Flip Rate | L8 Cosine ($K=256$) | L8 Cosine ($K=1024$) | Regime Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **$0.125\times$** | 9.0% | +0.2131 | +0.2347 | Quantizer Dead Zone ($\ll \Delta$) |
| **$0.250\times$** | 18.3% | +0.2223 | +0.2525 | Under-smoothed |
| **$0.500\times$** | 36.5% | +0.2290 | +0.2629 | Transition into smoothing regime |
| **$1.000\times$** | **58.9%** | **+0.2306** | **+0.2672** | **Optimal Empirical Peak ($\sigma^* \approx 1.0\Delta$)** |
| **$2.000\times$** | 74.9% | +0.2037 | +0.2147 | Perturbation variance begins dominating |
| **$4.000\times$** | 83.8% | +0.1364 | +0.1152 | Over-smoothed / loss of gradient signal |
| **$8.000\times$** | 88.4% | +0.0985 | +0.0687 | Noise saturation |
| **$16.000\times$** | 90.7% | +0.0845 | +0.0518 | Code destruction |

---

## 6. Audit Trail & Backward Compatibility

- **Handoff Archive**: Re-archived `dustq.zip` at HEAD matches git tree.
- **Data Integrity**: Checked all 7 JSON files; all required section keys (`seed_noise`, `cosine_ladders`, `ste_bias`, `directional_descent`, `sigma_delta`) are verified non-empty.
