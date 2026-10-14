# Hypotheses and analysis plan (pre-registered)

Committed **before any Tier B training run** (Plan v3.1 section 7, task T0-hyp; T6 probes
may run first, but no B-tier training result may inform a change to this file without an
explicit deviation note in the paper's appendix). Tags and rules of Plan v3.1 apply.

Frozen: 2026-10-11.

## Methods compared

Same model, data order, init and optimizer family at every cell (Plan v3.1 section 3.1):
**BP** (backprop, fp), **BP-STE** (quantized forward, identity STE on activation and
weight quantizers), **BP-SSTE** (activation-quantizer backward = derivative of the
Gaussian-smoothed quantizer E[q(x + sigma*e)] with sigma matched to Dust's per-site sigma;
weights keep identity STE), **Dust-K** (Dust with the same quantized forward; latent fp32
weights). Optional, Tier D only: BP-LSQ, ZO-W (weight-space ZO).

## Hypotheses

- **RQ1** — how does (Dust − BP-STE) change as activation bits drop (fp → A8 → A4 → A2)?
  - H1: at A8 the gap equals the fp gap within CI.
  - **H2 (primary): at A4 (W1.58A4), the interaction
    I = (Dust − BP-STE)|A4 − (Dust − BP)|fp is < 0.**
- **RQ2** — is any Dust advantage explained by smoothing?
  - H3 (exploratory, pre-stated direction): at A4, Dust − BP-SSTE < 0 (Dust beats analytic smoothing).
- **RQ3** — scaling with K (256 → 4k), size (S = L4/d256, M = L8/d512) and tokens (1M/10M).
  - H4: as bits drop, the cosine law's c_max falls and its c rises.
- **RQ4** — sigma–Delta mechanism.
  - H5: at A4/A2 the optimal sigma* is proportional to the median activation quantization step Delta,
    with a layer-type constant; at A8 sigma* is near the upstream literal.
- **RQ5** — STE-bias probe.
  - H6 (exploratory): cos(STE, Dust-16k reference) per bit width predicts the RQ1 gap.

## Primary endpoint (one, and only one confirmatory claim)

**I = [Dust-1k − BP-STE]@P3 − [Dust-1k − BP]@P0** on S/1M tokens (P3 = W1.58A4), test loss
at the best-validation checkpoint, paired by seed (upstream pairs make_model(seed) with
data order seed+1 for all methods). 95% CI by paired bootstrap, 10,000 resamples.
**H2 is claimed only if the CI excludes 0.** n = 5 seeds on the primary cell and its fp counterpart, unconditional (D2 is approved; a conditional n would allow optional stopping on the confirmatory endpoint).

**Key secondary:** [Dust-1k − BP-SSTE]@P3 (H3), same procedure, labelled exploratory.

## Everything else is descriptive or exploratory

Other K values, sizes, token budgets and bit widths: report point estimates with CIs and no
significance language. The trend over bits is reported as a fitted slope with a CI. Diverged
runs are scored at the best validation reached before divergence and flagged. No run is
dropped; deviations from this plan are listed in the paper's appendix.

## Probe gates that can end the confirmatory plan early (Plan v3.1 section 8)

- G2a fails (no Dust signal at P3): probe-only paper; H2/H3 are never tested with training.
- A4 shows no STE bias in probe 3 (cos(STE, ref) ≈ cos(fp-BP, fp-ref)): H2 unlikely; P4
  (W1.58A2) becomes primary if G2b passes, with this file updated and the change disclosed.
