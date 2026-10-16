# MONITORING — standing research & verification watch

The project's standing watch for scooping detection, adjacent-work updates, and optimization
leads. Any agent (or the owner) can execute it identically: follow the checklist below,
write the dated report, return the verdict. **Cadence** (enforced by the plan): before every
gate decision (G0/G2/G3...), monthly as a floor, and always before any submission or
preprint posting.

## Why this exists

`docs/related_work.md` (2026-10-11) found the niche open but flagged live risks: Q Labs
listed non-differentiable architectures as Dust's future work; FOGZO (NeurIPS 2025) does
STE+ZO quantization-aware *pretraining*; the QuZO/ZeroQAT/KronZO groups are active. No one
had cited Dust as of 2026-10-11. The watch turns that one-time scan into a standing tripwire.

## Checklist (each run)

1. **Upstream Dust repo** — fetch https://github.com/qlabs-eng/dust/commits/main and compare
   against our pin (`third_party/dust/UPSTREAM_COMMIT`, currently `b20f7c03eac...`). New
   commits matter doubly if they touch quantization, dtype, recipes, or bug fixes we
   reproduced around. Summarize any diff vs our pin.
2. **Citations of Dust** — search Semantic Scholar / Google Scholar / arXiv full-text for
   the report (qlabs.sh/research/dust) and its authors. Any citing paper is read at
   abstract level minimum; overlap with our niche (activation-space perturbation + low-bit
   forward + from-scratch pretraining) triggers an ACT verdict.
3. **New adjacent work** (listings since the previous watch date) — arXiv searches:
   "zeroth-order" + quantization; "node perturbation" / "activation perturbation" + quantized;
   quantized pretraining from scratch; ES/forward-only LLM training; BitNet-style follow-ups.
4. **Key groups** — new papers by: the Dust authors (Dahal, Mandal, Gülbahçar, Vegesna / Q
   Labs); the FOGZO authors; the QuZO group (Zhou et al.); Allaire/KronZO (GERAD); the
   ZeroQAT group; BitNet (Microsoft). Anything new: fetch and summarize.
5. **Optimization leads** (for our measured bottlenecks) — quantized-code caching /
   fake-quant speedups; integer matmul on Ada GPUs; torch.compile for small training loops;
   SDPA dense-mask costs; anything applicable to the 2.48–3.18× quant overhead and the
   ~2.5–2.8× minimal-code slowdown (docs/reports/T1.md). Cite sources; do not adopt
   anything without owner approval.
6. **Standing verification items** — re-check the open [unverified] items in
   `docs/OPEN_QUESTIONS.md` (currently: EGGROLL's integer-datatype experiment needs a
   full-text read; AGZO/ZO-Act need primary-source verification). Resolve what can be
   resolved with fetches.

## Output format — `docs/reports/WATCH/YYYY-MM-DD.md`

```
# Watch YYYY-MM-DD
Fetched: <date>; run by: <agent/owner>
Verdict: NONE | WATCH | ACT   (ACT = something overlaps our niche or invalidates a claim)
## Dust upstream      (new commits vs pin, or "unchanged")
## Citations of Dust  (new citing papers, or "none found")
## Adjacent work      (new papers with one-line "what it actually does")
## Optimization leads (applicable ideas with links)
## Verification      (items resolved / still open)
## Impact on plan     (one paragraph: what this run changes, if anything)
```

Rules: verify every claim with a fetch (never repeat a listing's abstract as fact without
reading it); tag [verified]/[unverified]; no plan changes without owner approval; the
watch report is committed in the same session it is produced.

## Dispatch template (paste to a fresh agent)

> You are the Dust-Q standing watch (protocol: docs/MONITORING.md — read it first). Today
> is <date>. Our pinned upstream is b20f7c03eac630b6441ba6f254128c1761dc45ad. Execute all
> six checklist items, write docs/reports/WATCH/<date>.md in the specified format, and
> report back a 10-line summary ending with the verdict. Context: we extend Dust (Q Labs,
> Oct 2026, zeroth-order transformer pretraining via per-token activation perturbation) to
> low-bit quantized forward passes trained from scratch (7M/38M params, int8/ternary
> weights, A8/A4 activations). Our unclaimed-niche claim and must-differentiate papers are
> in docs/related_work.md; top risks: FOGZO (arXiv:2510.23926), EGGROLL's integer-datatype
> experiment (arXiv:2511.16652), the QuZO family. Do not modify any other file.
