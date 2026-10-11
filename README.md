# Dust-Q

Backprop-free pretraining of low-bit transformers: can **Dust** (Q Labs' activation-space
perturbation method) pretrain **quantized** transformers (int8 / ternary weights, low-bit
activations) from scratch — and does measuring the real quantized loss beat backprop+STE
and a Gaussian-smoothed STE control at matched token budgets?

Working title: *Backprop-free pretraining of low-bit transformers via activation-space perturbation.*

## Where to look

| Question | Document |
| --- | --- |
| What is the plan? | `docs/planning/Dust-Q Implementation Plan v3.md` (rev 3.1 — **canonical**, supersedes v1/v2 in the same folder) |
| If I stop now, is it publishable, and where? | `docs/PUBLICATION_STAGES.md` (updated at every gate) |
| What is still unverified? | `docs/OPEN_QUESTIONS.md` |
| Where do upstream facts live? | `docs/REPO_MAP.md`, `docs/paper_tables.md`, `docs/related_work.md` |
| What was pre-registered? | `docs/HYPOTHESES.md` |
| Task reports | `docs/reports/T*.md` |

## Layout

```
third_party/dust/   upstream qlabs-eng/dust, vendored and pinned (MIT; UPSTREAM_COMMIT)
dustq/              our package: upstream.py, config.py, recipes.py, quant.py, qmodel.py,
                    dust_ext.py, backprop_ext.py, runner.py, train.py (resumable Runner),
                    queue.py (idempotent)  (+ probes.py, data.py later)
configs/            one YAML per run (scripts/make_configs.py --tier R)
scripts/            queue.py, benchmark.py, make_configs.py, check_data.py,
                    colab_t1.py (paste-into-Colab T1)  (+ noise_sweep.py, aggregate.py,
                    make_figures.py later)
results/            runs.csv, per-run logs, benchmarks, checkpoints (gitignored)
docs/               planning/, reports/, and the documents above
tests/              unit + acceptance tests (Phase 0 + T1 resume-exactness)
```

Upstream is never edited; all behaviour changes live in `dustq/` (subclass/extension per
Plan v3.1 section 4).

## Setup (Phase 0, CPU-only)

```
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tests -x -q
```

GPU training (Phase 1+) runs on Colab L4 (bf16); upstream's own CLI requires CUDA + Linux,
which is why the CPU acceptance path goes through `dustq.runner` (Plan v3.1 section 7).

## Maintenance rules

- Update `docs/PUBLICATION_STAGES.md` in the same commit as every gate recommendation.
- Update `docs/OPEN_QUESTIONS.md` as items resolve; never delete resolved items.
- One commit per task; each task ends with a report in `docs/reports/`.
- Never tune on the test set; log every run including failures (Plan v3.1 sections 9, 6.5).
