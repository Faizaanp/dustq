# CUT_CELLS — what the 2026-10-16 re-cut removed, what it costs to restore, and how

Living registry. The re-cut (approved by the owner 2026-10-16 after the Colab T1
measurements, docs/reports/T1.md) trimmed **descriptive cells only** — every confirmatory
cell (the primary endpoint and its fp counterpart at 5 seeds, BP-SSTE, the probes, and
Tier R/G0) is uncut, and `docs/HYPOTHESES.md` is frozen: restores never touch it.

## Restore policy (the owner's decision, recorded 2026-10-16)

**We will go back and restore these cells if** (a) more compute arrives (more/larger CU
allowance, better GPU time, or freed schedule), or (b) the findings are good enough to
warrant the breadth (e.g., G2a/G3 pass with a clear result, or reviewers ask for it).
The default vehicle is the journal extension (Plan v3.1 T12); earlier restores are fine
once the Stage-4 core is secured. Restored cells are labeled post-hoc where they appear
in the paper. Mechanics: drop the listed YAMLs into `configs/` and re-run
`scripts/run_queue.py` — the queue skips completed runs and only trains what is new.

## Registry (prices at 2026-10-16 measured rates, ×1.52 CU/L4-h)

| # | Cell | As originally specced | After the cut | Runs to restore | Price (L4-h / CU) | Restore priority |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | B-S fp K=4k (S/1M, dust) | 3 seeds | 1 seed (s0) | 2 | 5.6 / 8.6 | 1 — completes the fp K-ladder |
| 2 | Tier A P3 K=1k (S/10M, w1.58a4) | 3 seeds | 1 seed (s0) | 2 | 36.3 / 55 | 2 — 10M quantized at full CI |
| 3 | B-M K=1k (M/1M, P2+P3) | 3 seeds | 1 seed (s0) per precision | 4 | 25.6 / 39 | 3 |
| 4 | B-S quantized K=4k (P1–P4) | 3 seeds each | not run | 12 | 86.3 / 131 | 4 — journal version |
| 5 | D1 (M/10M, P0+P3, K256) | 3 seeds | 1 seed per precision | 4 | 43.9 / 67 | 5 — journal, gated |
| 6 | Tuning grid for P1/P2/P4 | 3-point × K{256,1k} | 2-point {1.0, 1.5}× | ~8 short runs | ~17 / 26 | 6 |
| 7 | Tier A P2 (S/10M, w1.58a8) | "+P2 if budget allows" | not planned | 6 | 67.2 / 102 | 7 — journal |

Core-restoration total (1–6): **~215 L4-h ≈ 327 CU** ≈ the gap to the uncut matrix
(~345 → ~175 L4-h as measured). The T5 code-caching fix will make quantized restores
cheaper (~1.3× fp instead of 2.5–3.2×). **Concurrency does not pay: measured 2026-10-16,
two concurrent S jobs ran ~2.9× slower than serial (116.1 s vs 40.6 s; 0.35×)** — the
minimal-code workload is CPU/kernel-launch-bound — so the prices above stand at sequential
rates, and buy-backs ride on T5 caching plus any additional compute that arrives.

## How to restore (example)

To restore registry #2 (Tier A P3 K=1k, seeds 1–2), create in `configs/`:

```yaml
task: A
size: L4d256
tokens: 10M
precision: w1.58a4
method: dust
population: 1024
seed: 1            # and a second file with seed: 2
data_dir: data
```

(lr/momentum/steps/eval_interval default to the upstream-tuned cell automatically).
Then run `python scripts/run_queue.py --configs-dir configs --results-dir <DRIVE results>`
— it trains only the new cells and appends them to runs.csv. All other restores are the
same pattern: the table above gives the exact field values (see `scripts/make_configs.py`
for the code path). The K=64 discrete-update pilot (D4) and D3/D5 extensions were never
cut — they were already conditional and stay as planned.
