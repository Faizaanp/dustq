# REPO_MAP — upstream qlabs-eng/dust at commit b20f7c03eac630b6441ba6f254128c1761dc45ad

Vendored at `third_party/dust/` (MIT; pin in `third_party/dust/UPSTREAM_COMMIT`). Line
numbers below refer to that commit. Second column facts were verified by two independent
reads (docs/OPEN_QUESTIONS.md items 13–17). Upstream is never edited; all our behaviour
lives in `dustq/`.

## Entry points

| What | Where | Notes |
| --- | --- | --- |
| DUST training CLI | `dust.py:478 main()` | requires CUDA (`dust.py:497`), refuses non-empty output dir, `--population` choices {256, 1024, 4096, 16384}, `--steps` smoke flag |
| Backprop baseline | `baselines/backprop.py:45 main()` | single GPU only (torchrun refused), requires CUDA |
| Data prep | `prepare_data.py` | skips 200,000 FineWeb docs, encodes a 20.5M-token pool (10,004 rows of 2049), saves 1M/10M prefixes (488/4880 rows) and 544-row val/test after the pool; idempotent (hash-checked saves); `manifest.json` with sha256s |
| Model | `model.py:113 GPT` | built via `dust.py:437 make_model` (meta-init, `to_empty`, then `init_weights` + c_proj re-init ±1.5·sqrt(3/fan_in), lm_head std 0.001, x0_lambdas 0.1) |

## Estimator structure (`dust.py`)

- `dust.py:104 DUST.__init__` — registers a forward hook on every `nn.Linear`/`nn.Embedding`
  (capture input/output, apply output jitter) **and** a pre-hook on each `attn.c_proj`
  (the virtual sites `o@{layer}`, `dust.py:130`). This is where our `QLinear` swap must stay
  `nn.Linear`-compatible and where hook *registration order* matters.
- `dust.py:138 output_hook` — captures `args[0].float()` into `self.inputs` during the clean
  pass; pre-hook replacements (our dequantized activations) are visible to it. **G_W
  therefore uses the quantized-dequantized input automatically** (Plan v3.1 T5 test 4).
- `dust.py:175 direct_errors` — full-forward draws in chunks of `nrep`; **no prefix
  caching**; reward = clean − perturbed loss, centered per chunk; `error -= einsum(reward, noise)/sigma`.
- `dust.py:201 head_error` — jitters cached logits one 256-column slab at a time; soft-cap
  `15*tanh(x/15)` repeated; normalized by `draws // slabs`.
- `dust.py:245 local_errors` — attention internals (q, k, v, gate, ve) recomputed **per
  block from cached clean c_q/c_k/c_v outputs** (rotary + QK-norm reapplied identically),
  scored against the `o@l` error (a linear surrogate) with gamma-decayed credit from later
  tokens for all kinds except q. **These credits never touch downstream activation
  quantizers** — which is what makes our 4-site activation quantization safe.
- `dust.py:333 scalar_gradients` — two-sided weight-space ES for `resid_lambdas`/`x0_lambdas`;
  8 draws = **16 full forwards per step** (counted in T1's cost model, not in K).
- `dust.py:325 reassemble` — `G_W = einsum('btd,bti->di', error, input)`; embedding via index_add.
- `dust.py:354 step()` — **all noise scales are literals here**: writers/hidden/wte 0.2;
  `o@l` 0.2 shallow half / 0.4 deep half; head 0.05; local 0.05; scalars 0.03.
  `dustq/dust_ext.py::DUSTQ` is a copy with these replaced by `self.sigma` (bitwise-tested).
- Non-finite estimates raise (`dust.py:387`); gradients are written to `.grad` and
  `torch.optim.SGD.step()` runs; summed estimates are divided by the valid-token count.
- Noise RNG: `dust.py:118` — device-resident `torch.Generator`, seeded `seed + 3 + 7777·rank`.
  **Resume must save/restore its state** (`DUSTQ.state_dict` does).
- Pairing: `make_model(config, device, seed)` + data order `torch.Generator().manual_seed(seed + 1)`
  (`dust.py:509`) — identical for Dust and backprop → paired statistics are valid by construction.

## Cost structure (drives the Plan v3.1 section 6 budget)

- Full forwards per step ≈ K (direct draws) + 16 (scalars); head and local draws are cheap.
- Attention (`model.py:9`) is SDPA over a dense boolean mask — full T×T even for half-window
  blocks. Window pattern 'SSSL' (`model.py:27`): in L8, blocks 3 and 7 are full-window.

## Optimizers (asymmetric — preserve per-method)

- Dust `dust.py:453 make_optimizer` — groups: wte lr 1000, value_embeds lr 0.3,
  lambdas lr 0.03, everything else `settings.lr`.
- Backprop `baselines/backprop.py:22 make_optimizer` — groups: wte lr 1000, **everything
  else at base lr** (no special lr for value_embeds/lambdas).
- Backprop defaults `baselines/backprop.py:16` — 1M: lr 0.175, momentum 0.98 (61 steps);
  10M: lr 0.2, momentum 0.95 (610 steps).

## Dtype facts

- `dust.py:117` `self.dtype = bfloat16 if cuda else float32` — instance attribute, overridable post-construction (no upstream edit needed).
- `model.py:141 init_weights` casts wte and value_embeds to bf16 **on CUDA only** — a module
  dtype change, not `self.dtype`; fp32 runs must cast them back (G1, T4).
- `dust.py:422 evaluate()` hard-codes bf16 autocast on CUDA (ignores `self.dtype`) → G1's
  fp32 comparisons must use our own evaluation path.
- Both entry points set `torch.set_float32_matmul_precision('high')` (TF32) → T4 must fix
  and log a TF32 policy.

## Data

- `load_sequences` (`dust.py:400`) expects [sequences, 2049] int tensor, ids in [0, 4096);
  training subset = first `steps × 8` rows, shuffled by `seed + 1`.
- One epoch; 61 steps @ 1M (488 rows), 610 @ 10M (4880 rows). A 20M budget = 9,768 rows
  (steps=1221), a strict prefix of the 20.5M pool — one-line extension in `dustq/data.py`.

## Where our quantization hooks go (Plan v3.1 section 4.2)

- Weight quant: swap `attn.c_q, attn.c_k, attn.c_v, attn.c_proj, mlp.c_fc, mlp.c_proj` →
  `dustq/quant.py::QLinear` (an `nn.Linear` subclass, so capture/optimizer groups are unchanged).
- Activation quant: pre-hooks on the inputs of those same six Linears
  (`dustq/qmodel.py::register_act_hooks`); unsigned grid for `mlp.c_proj` (post-ReLU²).
  **Register AFTER DUSTQ construction** → noise → quantize at `o@l`.
- Never quantized: q/k after rotary+norm, attention probabilities, ve_gate, embeddings,
  value embeddings, LM head and its input, norms, λ scalars (keeps `local_errors` and
  `head_error` exact).

## Reproduction targets (test loss @ best val)

- M (L8/d512) targets are from **Appendix C** (docs/paper_tables.md): 1M — Dust 256/1k/4k = 6.067/5.934/5.911, backprop 5.959; 10M — Dust 256/1k = 5.231/5.111, backprop 4.989.
- S (L4/d256) @ 10M is from **main-post Figure 4** (the size-study table; it predates the main protocol — see Plan v3.1 T3 for the caveat): backprop 5.066; Dust 64: 5.556, 256: 5.362, 1k: 5.161, 4k: 5.095, 16k: 5.065 [paper].
- G0 pass: M/1M within 0.05 of Appendix C per K, same K ordering; S/10M/K256 within 0.08 of 5.362.

## Our format deviations (dustq, not upstream)

- `results/<run_id>/best.pt` is `{"best": float, "state": state_dict}` — atomic, self-describing, read ONLY via `dustq.train.load_best`. Upstream saves a bare state_dict; do not load ours with a plain `torch.load` + `load_state_dict`.
- The queue CLI is `scripts/run_queue.py` (a literal `scripts/queue.py` shadows the stdlib `queue` module that torch imports).
