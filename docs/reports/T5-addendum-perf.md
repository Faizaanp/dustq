# T5 addendum — the exact integer path measured slower, then fixed

Session 2026-10-31, Colab L4, post-T5 benchmark (`--precisions fp,w1.58a4`; 1 warmup + 2
measured steps per cell). Full table: docs/reports/WATCH or colab log; key rows:

| cell | pre-T5 fake-quant | post-T5 exact (fp32 matmul) | fp |
| --- | --- | --- | --- |
| S K256 W1.58A4 | 25.135 | 35.270 | 10.103 |
| S K1024 W1.58A4 | 107.049 | 149.963 | 42.007 |
| M K256 W1.58A4 | 98.269 | 149.732 | 31.214 |
| M K1024 W1.58A4 | 377.431 | 573.944 | 118.859 |

The exact path was NOT the budget win the re-cut hoped for: **~1.5x slower than the
pre-T5 fake-quant path, ~1.85x fp instead of the targeted ~1.3x**, peak memory +~12%.

## Root cause

The first implementation computed the code matmul in **fp32 with TF32 disabled** — i.e.
FP32 CUDA cores, no tensor cores. The pre-T5 path cast dequantized values to bf16 and ran
bf16 tensor cores; its per-draw weight re-quantization cost was real but secondary.
Under autocast, PyTorch defaults to **fp32_precision = ieee** for the CUDA backend
quoted in its own docs, so `@` on fp32 tensors runs without tensor cores.

## Fix (attribute, not patch)

Codes are small integers: weights ∈ {-1,0,1}·γ or {−128..127}, activations likewise.
TF32 keeps 11 significand bits, so every input and product is exact and accumulation is
fp32 — the result is **bit-identical to the fp32 path** but runs on tensor cores.
Implementation: `dustq/quant.py::_tf32_int_matmuls` saves the current precision setting,
sets `allow_tf32=True` for one matmul, restores in `finally`. No other path is affected;
`probe_averaging_test` etc. still compare cleanly.

## Honest budget consequence

The re-cut's "1.3x fp after T5" was optimistic; the corrected expectation is
"**~1.3-1.5x fp with the TF32 path, matching pre-T5 fake-quant**" — pending the
next benchmark run with `--exact-mode tf32,fp32,fake` (all three rows, one job, so
the attribution question is settled by data, not argument). CUT_CELLS restore prices
stay at the PRE-T5 (fake-quant) numbers until that table lands.

## What this addendum does not change

- The exactness property (fp64 reference within 1e-7, integer code path bit-exact).
- The Dust-only scope of the exact path (BP-STE/BP-SSTE keep fake-quant for train/eval
  self-consistency, deliberately).
- Any claim in the paper: Dust and BP-SSTE see the same quantized forward either way.
