# Post-Fix Science Report: T6 Probe Results & Honest Gate G2a Characterization

> **Rule-6 Dated Update (2026-11-22 — Clean Schema v2 Re-Run & Reframing):** The 2026-11-20 code fixes (`schema_version = 2`, independent ladder rung seeds $\ge 10000$, `MIN_RELIABILITY = 0.2` guard, disjoint validation selection `val[:8]` vs 4-batch held-out reporting `val[8:40]`, multi-seed Dust candidates, global `norm` logging) have been **re-run and verified on Colab L4** across Small ($S$, $L4d256$, $P0$–$P3$) and Medium ($M$, $L8d512$, $P3$).
>
> Crucially, the clean Medium dataset **refutes any "Dust beats STE by 45–54%" headline**: on Medium, `Dust1k` (`-0.0087`), `Dust256` (`-0.0099`), `STE` (`-0.0082`), and `SSTE` (`-0.0085`) are all sub-noise within $\approx 0.008$–$0.010$ nats (`Dust1k` 3-seed std is `0.0023`, ~26% of its mean), and Small's apparent edge depended on re-tuning away from a failing train step size (`+0.0669` overshoot). What survives cleanly across **both** model sizes is the **step-size transfer gap (`STE < SSTE < Dust` in `val_at_train_eps` failure)** and the **$\sigma/\Delta$ flip-rate characterization**. All sections below (including §§**3.1** and **3.2**) report canonical **2026-11-22 Schema v2** data alongside dated Rule-6 historical notes.

**Date**: 2026-11-14 (Post-Fix Cohort); **Updated 2026-11-22 (Audit-Clean Schema v2 Cohort)** `[code]`  
**Governing Documents**: `AGENTS.md`, `docs/planning/Dust-Q Implementation Plan v3.md` (v3.1, §§5, 7, 8), `docs/reports/T6_AUDIT_2026-11-14.md` `[design]`  
**Hardware Platform**: Google Colab NVIDIA L4 GPU (burn rate: 1.94 CU/h) `[code]`  
**Target Precision**: $W1.58A4$ (Plan v3.1 Gate Precision P3) `[design]`  
**Canonical Schema v2 Probe Artifacts (`results/probes/`)**:
1. `L4d256_1M_w1p58a4_backprop_K0_s0_probes_0-2-3.json` (`schema_version: 2`, Probes 0, 1, 2, 3 on $S$ $P3$) `[code]`
2. `L4d256_1M_w1p58a4_backprop_K0_s0_probes_4.json` (`schema_version: 2`, disjoint 4-batch / 3-seed Probe 4 on $S$ $P3$) `[code]`
3. `L4d256_1M_w1p58a4_backprop_K0_s0_probes_5.json` (16-point $\sigma$-$\Delta$ sweep on $S$ $P3$; `r_g_median` corrected to `0.339648`) `[code]`
4. `L4d256_1M_fp_backprop_K0_s0_probes_2-3.json` (`schema_version: 2`, $S$ $P0$ baseline) `[code]`
5. `L4d256_1M_w1p58a8_backprop_K0_s0_probes_2-3.json` (`schema_version: 2`, $S$ $P2$ baseline) `[code]`
6. `L4d256_1M_w8a8_backprop_K0_s0_probes_2-3.json` (`schema_version: 2`, $S$ $P1$ baseline) `[code]`
7. `L8d512_1M_w1p58a4_dust_K256_s0_fake_G1_probes_2-5.json` (`schema_version: 2`, $M$ $P3$ ladders + 16-point $\sigma$-$\Delta$ sweep) `[code]`
8. `L8d512_1M_w1p58a4_dust_K256_s0_fake_G1_probes_4.json` (`schema_version: 2`, disjoint 4-batch / 3-seed Probe 4 on $M$ $P3$) `[code]`

---

## 1. Executive Summary & Honest Gate G2a Verdict

| Component | Honest Verdict (`Schema v2`, 2026-11-22) | Evidence Summary |
| :--- | :--- | :--- |
| **Arm 1 (Cosine Ladders)** | **PASS on Small ($P0$–$P3$); PARTIAL / BELOW-FLOOR on Medium (`MLP in`)** `[code]` | Small passes strictly across all 5 writer/hub groups (`r_g_median = 0.3396`). Medium passes on `2/3` MLP+writer criterion groups (`Attn out` `r_g=0.745`, `MLP out` `r_g=0.446`) plus `Head`/`Tok emb`, while `MLP in` falls below the `MIN_RELIABILITY = 0.2` floor (`r_g = 0.1298` $\to$ `None` disattenuated, raw `0.0343 -> 0.0927 -> 0.1724`; overall 12-group median `r_g = 0.1070`). |
| **Arm 2 (Directional Descent)** | **"Beats STE" = NULL; literal $\Delta L_q < 0$ PASSES** `[code]` | Descent exists everywhere ($\Delta L_q < 0$ on all 12/12 batch-seed evaluations for both Small and Medium), **but `Dust1k ≈ STE ≈ SSTE` on Medium** (`-0.0087` vs `-0.0082` vs `-0.0085`; `Dust1k` seed std `0.0023` $\approx 26\%$ of its mean). Any "beats STE" claim is null. |
| **The Surviving Result** | **Step-Size Transfer Gap (`STE < SSTE < Dust`) + $\sigma/\Delta$ Flip-Rate Mechanism** `[code]` | **Plain STE transfers** its train step $\varepsilon_{\text{train}}^*$ cleanly to validation on both Small (`-0.0110`) and Medium (`-0.0055`). **SSTE** transfers on Small (`-0.0117`) but **fails on Medium** (`+0.0313` at $\varepsilon=0.4$). **Dust** fails step transfer on **both** Small (`+0.0588..+0.0669` at $K=1\text{k}$) and Medium (`+0.0108..+0.0575`) across every seed. |

Plan v3.1 §7 line 310 sets the literal criteria for Gate G2a:
> *"G2a: at P3, cos(Dust-1k, ref) rises with K on MLP and writer layers, and directional descent along Dust-1k is negative. G2b: the same at P4."* `[paper]`

Literally, Small ($L4d256$) passes both Arm 1 and Arm 2, and Medium ($L8d512$) shows $\Delta L_q < 0$ across all held-out batches and seeds. Scientifically, however, the gate establishes **equivalence (`Dust ≈ STE ≈ SSTE`) accompanied by an estimator-noise step-transfer penalty**, not ZO superiority `[code]`.

---

## 2. Gate G2a Evaluation Matrix (Schema v2 — 2026-11-22)

| Criterion | Architecture | Plan v3.1 Spec | Schema v2 Measured Reality (2026-11-22) | Honest Status |
| :--- | :--- | :--- | :--- | :---: |
| **Arm 1: Writer/MLP Ladder Rise** | Small ($L4d256, W1.58A4$) | Strictly rising with $K \in \{256, 1024, 4096\}$ | `Attn out`: $0.4610 \to 0.6705 \to 0.7851$ (dis $0.8405$)<br>`MLP out`: $0.2798 \to 0.5278 \to 0.6701$ (dis $0.7610$)<br>`MLP in`: $0.1454 \to 0.3247 \to 0.4631$ (dis $0.6033$)<br>`Head`: $0.6869 \to 0.8750 \to 0.9517$ (dis $0.9676$)<br>`Tok emb`: $0.1603 \to 0.3731 \to 0.5834$ (dis $0.7112$) | **PASS** `[code]` |
| **Arm 1: Writer/MLP Ladder Rise** | Medium ($L8d512, W1.58A4$) | Same as above | **2/3 criterion groups above $r_g \ge 0.2$ floor:**<br>`Attn out`: $0.3032 \to 0.5975 \to 0.7902$ (dis $0.8551$, $r_g=0.745$)<br>`MLP out`: $0.1486 \to 0.3291 \to 0.5256$ (dis $0.6692$, $r_g=0.446$)<br>**`MLP in` below floor:** raw $0.0343 \to 0.0927 \to 0.1724$ ($r_g=0.1298 \to \text{None}$)<br>Median $r_g$ across 12 groups = **$0.1070$** | **PARTIAL / BELOW FLOOR ON `MLP in`** `[code]` |
| **Arm 2: Directional Descent ($\Delta L_q < 0$)** | Small & Medium ($P3$) | $\Delta L_q(\text{train}) < 0$ and $\Delta L_q(\text{val}) < 0$ | Small `Dust1k`: train $-0.1105$, held-out val **$-0.0170 \pm 0.0003$**<br>Medium `Dust1k`: train $-0.0687$, held-out val **$-0.0087 \pm 0.0013$** | **PASS (literal)** `[code]` |
| **Arm 2: Direction Superiority vs STE** | Small & Medium ($P3$) | Does Dust beat STE/SSTE? | Medium: `Dust1k` ($-0.0087$) $\approx$ `SSTE` ($-0.0085$) $\approx$ `STE` ($-0.0082$)<br>Small gap ($-0.0170$ vs $-0.0110$) requires val $\varepsilon$ re-tuning | **NULL** `[code]` |
| **Step-Size Transfer (`val_at_train_eps`)** | Small & Medium ($P3$) | Does $\varepsilon_{\text{train}}^*$ descend on held-out val? | **Rank order `STE < SSTE < Dust`:**<br>`STE`: passes S (`-0.0110`) & M (`-0.0055`)<br>`SSTE`: passes S (`-0.0117`), fails M (`+0.0313`)<br>`Dust`: fails S (`+0.059..+0.067`) & M (`+0.011..+0.057`) | **CONFIRMED MECHANISM** `[code]` |

*(Rule-6 Historical Note: On 2026-11-14 before A1/A2/A5 fixes, raw Arm 1 at $K=4096$ shared `seed=0` with reference half `g1`, inflating `disattenuated` to `1.0000`, and `T6_POST_FIX_RESULTS` previously flattened Medium Arm 1 into an unqualified pass.)*

---

## 3. Directional Descent & $\varepsilon$-Transfer Analysis (Probe 4 — Schema v2, 2026-11-22)

Each candidate is stepped under **global Euclidean parameter-vector norm** `[code]`: $\varepsilon_{\text{train}}^*$ is selected on `ctx.x, ctx.y`; $\varepsilon_{\text{val}}^*$ is selected on `val[:8]` (`val_selected_in_batch`); unbiased held-out $\Delta L_{\text{val}}$ is measured across **4 disjoint held-out batches (`val[8:40]`)**; and **`val_at_train_eps`** applies the train-optimal step $\varepsilon_{\text{train}}^*$ directly to `val[8:40]` to test step-size transfer without re-tuning `[design]`.

### Cross-Scale $\varepsilon$-Transfer Summary (Key Table) `[code]`

| Model | Estimator | Train $\varepsilon^*$ | Val $\varepsilon^*$ | Held-Out `val ΔL` (`val[8:40]`) | `val ΔL` at Train $\varepsilon^*$ (`val_at_train_eps`) | Transfers? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Small ($L4$)** | **STE** | $0.80$ | $0.80$ | $-0.0110 \pm 0.0025$ | **$-0.0110$** | **YES** |
| **Small ($L4$)** | **SSTE** | $0.80$ | $0.80$ | $-0.0117 \pm 0.0025$ | **$-0.0117$** | **YES** |
| **Small ($L4$)** | **Dust256** (3 seeds) | $1.60$ | $0.80$ | $-0.0216 \pm 0.0002$ | **$+0.0012$ to $+0.0041$** | **FAIL** |
| **Small ($L4$)** | **Dust1k** (3 seeds) | $1.60$ | $0.80$ | $-0.0170 \pm 0.0003$ | **$+0.0588$ to $+0.0669$** | **FAIL** |
| **Medium ($L8$)** | **STE** | $0.20$ | $0.10$ | $-0.0082 \pm 0.0012$ | **$-0.0055$** | **YES** |
| **Medium ($L8$)** | **SSTE** | $0.40$ | $0.10$ | $-0.0085 \pm 0.0012$ | **$+0.0313$** | **FAIL** |
| **Medium ($L8$)** | **Dust256** (3 seeds) | $3.20$ | $0.80$ | $-0.0099 \pm 0.0005$ | **$+0.0472$ to $+0.0575$** | **FAIL** |
| **Medium ($L8$)** | **Dust1k** (3 seeds) | $1.60 / 0.80$ | $0.40 / 0.20$ | $-0.0087 \pm 0.0013$ | **$+0.0108$ to $+0.0442$** | **FAIL** |

### 3.1 Small Architecture ($L4d256, W1.58A4$) — Per-Seed Breakdown
*Source*: [`results/probes/L4d256_1M_w1p58a4_backprop_K0_s0_probes_4.json`](file:///E:/project/some%20research%20work%202/results/probes/L4d256_1M_w1p58a4_backprop_K0_s0_probes_4.json) (`schema_version: 2`, `val_report_n = 4`) `[code]`

| Estimator Candidate | $\varepsilon_{\text{train}}^* / \varepsilon_{\text{val}}^*$ | $\Delta L_{\text{train}}$ | Held-Out $\Delta L_{\text{val}} \pm \text{SE}$ (4 batches) | `val_selected_in_batch` | `val_at_train_eps` | Global Norm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard STE** | $0.80 / 0.80$ | $-0.119763$ | $-0.011004 \pm 0.002502$ | $-0.032783$ | **$-0.011004$** | $0.2653$ |
| **Smoothed STE (SSTE)** | $0.80 / 0.80$ | $-0.120729$ | $-0.011744 \pm 0.002549$ | $-0.033877$ | **$-0.011744$** | $0.2672$ |
| **Dust256 ($s=0$)** | $1.60 / 0.80$ | $-0.133451$ | $-0.021808 \pm 0.001390$ | $-0.031891$ | $+0.001211$ | $0.3590$ |
| **Dust256 ($s=1$)** | $1.60 / 0.80$ | $-0.130919$ | $-0.021249 \pm 0.001715$ | $-0.034228$ | $+0.003624$ | $0.3597$ |
| **Dust256 ($s=2$)** | $1.60 / 0.80$ | $-0.131136$ | $-0.021640 \pm 0.001846$ | $-0.033825$ | $+0.004107$ | $0.3599$ |
| **Dust256 (3-seed mean)** | — | — | $-0.021566 \pm 0.000166$ (std `0.000287`) | — | — | — |
| **Dust1k ($s=0$)** | $1.60 / 0.80$ | $-0.110453$ | $-0.016442 \pm 0.002275$ | $-0.034824$ | **$+0.066864$** | $0.2774$ |
| **Dust1k ($s=1$)** | $1.60 / 0.80$ | $-0.118082$ | $-0.016863 \pm 0.002198$ | $-0.034555$ | **$+0.060421$** | $0.2759$ |
| **Dust1k ($s=2$)** | $1.60 / 0.80$ | $-0.118883$ | $-0.017587 \pm 0.002330$ | $-0.035033$ | **$+0.058816$** | $0.2750$ |
| **Dust1k (3-seed mean)** | — | — | $-0.016964 \pm 0.000334$ (std `0.000579`) | — | — | — |

- *(Rule-6 Historical Note, 2026-11-14): The superseded 2026-11-14 table reported only the in-batch `val[:8]` minima (`STE: -0.032783`, `SSTE: -0.033808`, `Dust256: -0.031891`, `Dust1k: -0.034824`, gap `-0.002041`) without `val[8:40]`, multi-seed uncertainty, or `val_at_train_eps`.*

### 3.2 Medium Architecture ($L8d512, W1.58A4$) — Per-Seed Breakdown
*Source*: [`results/probes/L8d512_1M_w1p58a4_dust_K256_s0_fake_G1_probes_4.json`](file:///E:/project/some%20research%20work%202/results/probes/L8d512_1M_w1p58a4_dust_K256_s0_fake_G1_probes_4.json) (`schema_version: 2`, `val_report_n = 4`) `[code]`

| Estimator Candidate | $\varepsilon_{\text{train}}^* / \varepsilon_{\text{val}}^*$ | $\Delta L_{\text{train}}$ | Held-Out $\Delta L_{\text{val}} \pm \text{SE}$ (4 batches) | `val_selected_in_batch` | `val_at_train_eps` | Global Norm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard STE** | $0.20 / 0.10$ | $-0.043916$ | $-0.008161 \pm 0.001212$ | $-0.004621$ | **$-0.005488$** | $0.4926$ |
| **Smoothed STE (SSTE)** | $0.40 / 0.10$ | $-0.044831$ | $-0.008455 \pm 0.001224$ | $-0.005429$ | **$+0.031335$** | $0.4938$ |
| **Dust256 ($s=0$)** | $3.20 / 0.80$ | $-0.069611$ | $-0.009693 \pm 0.001687$ | $-0.006587$ | $+0.057450$ | $1.7503$ |
| **Dust256 ($s=1$)** | $3.20 / 0.80$ | $-0.077871$ | $-0.010871 \pm 0.001314$ | $-0.006689$ | $+0.047198$ | $1.7414$ |
| **Dust256 ($s=2$)** | $3.20 / 0.80$ | $-0.068403$ | $-0.009193 \pm 0.001459$ | $-0.006359$ | $+0.056993$ | $1.7487$ |
| **Dust256 (3-seed mean)** | — | — | $-0.009919 \pm 0.000497$ (std `0.000862`) | — | — | — |
| **Dust1k ($s=0$)** | $1.60 / 0.40$ | $-0.068695$ | $-0.011295 \pm 0.001401$ | $-0.007616$ | **$+0.044168$** | $1.0474$ |
| **Dust1k ($s=1$)** | $0.80 / 0.20$ | $-0.041584$ | $-0.007203 \pm 0.001163$ | $-0.003493$ | **$+0.019928$** | $1.0932$ |
| **Dust1k ($s=2$)** | $0.80 / 0.20$ | $-0.048788$ | $-0.007469 \pm 0.001012$ | $-0.004784$ | **$+0.010793$** | $1.1030$ |
| **Dust1k (3-seed mean)** | — | — | $-0.008656 \pm 0.001322$ (std **`0.002290`**) | — | — | — |

- *(Rule-6 Historical Note, 2026-11-14): The superseded 2026-11-14 table reported only single-seed `val_selected_in_batch` (`STE: -0.005108`, `SSTE: -0.005645`, `Dust256: -0.006587`, `Dust1k: -0.007616`, falsely framed as "`49.1%` deeper"). Under Schema v2, `Dust1k` ($-0.0087$, seed std $0.0023$), `Dust256` ($-0.0099$), `STE` ($-0.0082$), and `SSTE` ($-0.0085$) are statistically indistinguishable on Medium.*

---

## 4. Key Scientific Insights (Updated 2026-11-22)

1. **The $\varepsilon$-Transfer Gap Scales Monotonically with Estimator Noisiness (`STE < SSTE < Dust`)**:
   Forward-only and smoothed gradient estimates over-fit their step size on the training batch and overshoot held-out validation loss when applied at $\varepsilon_{\text{train}}^*$. Plain `STE` is the **only** estimator whose train-optimal step size transfers cleanly to validation on both Small (`-0.0110`) and Medium (`-0.0055`), whereas `SSTE` fails transfer on Medium (`+0.0313`) and `Dust` fails transfer on **both** Small (`+0.0012..+0.0669`) and Medium (`+0.0108..+0.0575`) across every seed `[code]`.
2. **No Evidence of ZO Direction Superiority (`Dust ≈ STE ≈ SSTE` on Medium)**:
   Once in-batch $\varepsilon^*$ selection bias and depth effects are accounted for, `Dust1k` (`-0.0087`, seed std `0.0023`) is within noise of `STE` (`-0.0082`) and `SSTE` (`-0.0085`) on Medium ($L8d512$). Even on Small ($L4d256$), Dust's apparent gap depends on re-tuning $\varepsilon$ from `1.60` down to `0.80` on validation rather than cleaner step-size transfer `[code]`.
3. **Reliability Attenuation at Depth (`r_g` Drop on Medium `MLP in`)**:
   Increasing depth ($L4 \to L8$) lowers split-half reference reliability (`median r_g`: $0.3396 \to 0.1070$), pushing `MLP in` below the `MIN_RELIABILITY = 0.2` disattenuation floor (`0.1298`) even though its raw cosine continues to rise with $K$ (`0.0343 -> 0.0927 -> 0.1724`) `[code]`.
4. **Quantized Dead-Zone Removal ($\sigma/\Delta$ Flip-Rate Sweep)**:
   Continuous pre-quantization flip-rate measurements (`probes_5.json` on $S$; `probes_2-5.json` on $M$) trace the complete smoothing transition from $\sim 9$–$10\%$ bin flips at $0.125\times\Delta$ through $59$–$63\%$ at $1.0\times\Delta$ up to $>90\%$ noise saturation at $16\times\Delta$ `[code]`.

---

## 5. Audit Trail & Reproducibility

- All raw Schema v2 JSON files reside on Google Drive at `MyDrive/dustq/results/probes/` and are synchronized locally in `results/probes/` `[code]`.
- Metadata fix applied 2026-11-22: corrected `L4d256_1M_w1p58a4_backprop_K0_s0_probes_5.json` `reference_meta.r_g_median` from the off-by-one value `0.417564` to the true 12-group median `0.339648`, and removed legacy pre-Schema-v2 artifacts (`probes_4_fixed.json` and `results/probes/corrected/`) `[code]`.
- `disattenuation` divisors ($\sqrt{2 r_g / (1 + r_g)}$) and `MIN_RELIABILITY = 0.2` guards verified exact to `1e-9` across all Schema v2 JSONs `[code]`.
- Automated test suite [`tests/test_t5.py`](file:///E:/project/some%20research%20work%202/tests/test_t5.py) passes 17/17 tests `[code]`.
- No model execution or calibration took place on the local host machine, strictly complying with `AGENTS.md` Rule 5 `[design]`.
