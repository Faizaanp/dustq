# Dust-Q: Backprop-Free Pretraining of Low-Bit Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-27%20passed%20%7C%204%20CUDA--skipped-brightgreen.svg)](tests/)
[![Target: TMLR](https://img.shields.io/badge/target%20venue-TMLR-purple.svg)](docs/PUBLICATION_STAGES.md)

**Dust-Q** is an empirical and theoretical investigation into training quantized autoregressive Transformers from scratch **without backpropagation**. It explores whether **Dust**—an activation-space zeroth-order perturbation method introduced by Q Labs—can overcome the notorious gradient-bias problems of the Straight-Through Estimator (STE) when pretraining models with ternary weights (BitNet $b1.58$) and low-bit activations ($A8$, $A4$, $A2$).

---

## Overview

### The Scientific Question
Traditional Quantization-Aware Training (QAT) relies on **Backpropagation with the Straight-Through Estimator (BP-STE)**. Because rounding and clipping operators have zero gradient almost everywhere, STE replaces their derivative with an identity pass-through:

$$\frac{\partial Q(x)}{\partial x} \approx 1$$

At 8-bit precision, this approximation is relatively benign. However, as activation bit-width drops to 4-bit and 2-bit, the STE gradient diverges significantly from the true directional loss gradient, causing training instability or catastrophic failure.

**Dust-Q asks:**
> *Can zeroth-order activation-space perturbations train low-bit transformers from scratch, and does measuring the real quantized loss landscape beat backprop+STE at matched token budgets?*

Because Dust computes parameter updates by injecting structured activation perturbations $\delta a$ and measuring the **actual scalar loss differences** $\Delta \mathcal{L}$, it directly queries the true discrete landscape without backpropagating through non-differentiable staircases.

```
       Clean Forward:        x ───> [ Layer l ] ─── a ───────────────> Q(a) ───> Loss(a)
                                                      │
       Dust Perturbation:    x ───> [ Layer l ] ─── (a + δa) ────────> Q(a + δa) ─> Loss(a + δa)
                                                                                  │
       Scalar Difference:    ΔL = Loss(a + δa) - Loss(a) <────────────────────────┘
       Weight Update:        ΔW ∝ einsum(ΔL · δa, x)   (No backprop pass, true quantized loss)
```

---

## Core Hypotheses & Baseline Controls

Dust-Q pre-registers a formal confirmatory testing protocol to distinguish whether any empirical advantage comes from **actual loss evaluation** or simply **implicit noise smoothing**:

1. **Primary Endpoint ($H_2$)**:
   $$\mathcal{I} = [(\text{Dust-1k} - \text{BP-STE}) \mid W1.58A4] - [(\text{Dust-1k} - \text{BP}) \mid \text{fp}] < 0$$
   Evaluated at matched token budget ($1\text{M}$ tokens, $S$ size) with $n=5$ paired seeds. Tested via $10,000$ paired bootstrap resamples.
2. **Mechanistic Control — Gaussian-Smoothed STE (BP-SSTE)**:
   Analytically smooths the activation quantizer during backpropagation:
   $$\mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2)} [Q(x + \epsilon)]$$
   - If **Dust beats BP-STE and ties BP-SSTE**: Dust's benefit is explained by implicit smoothing, which can be recovered analytically at $1/K$ of the compute cost.
   - If **Dust beats both BP-STE and BP-SSTE**: Evaluating the true non-convex quantized loss landscape provides fundamental optimization benefits beyond smoothing.
   - If **Dust loses or ties BP-STE**: Characterizes the efficiency boundary of activation-space zeroth-order optimization at low bit widths.

---

## Key Features & Design Principles

- **Ternary Weights ($W1.58$)**: BitNet $b1.58$-style weights $W \in \{-\gamma, 0, \gamma\}$ with latent $\text{fp32}$ master weights updated by the optimizer.
- **Low-Bit Activations ($A8, A4, A2$)**: Per-token scaling, unsigned grids for post-$\text{ReLU}^2$ activations (`mlp.c_proj`), and signed grids for attention projection inputs.
- **Noise-Before-Quantize Hook Order**: In Dust-Q, perturbation hooks fire before quantization ($o = Q(a + \delta a)$), ensuring perturbations directly probe quantization bin transitions.
- **Exact Bitwise Parity**: DUSTQ is an exact extension of upstream Dust; on full-precision models with identical seeds, it produces bitwise-identical noise, gradients, and loss trajectories.
- **Bitwise Deterministic Resumption**: Full state persistence (model parameters, optimizer momentum, DUSTQ PRNG state, and data order cursor) ensures resumed runs bitwise-match continuous runs on CPU, and match within numerical tolerance on CUDA.
- **Atomic Single-File Checkpoints**: Robust against unexpected crashes or disconnects during cloud training.

---

## Repository Structure

```
dust-q/
├── dustq/                  # Core library
│   ├── config.py           # Model architectures (S, M, etc.) and RunConfig resolver
│   ├── quant.py            # QLinear (ternary/int8) and ste_act_quantize
│   ├── qmodel.py           # Precision parsing (W1.58A4), module swapping, hooks
│   ├── recipes.py          # Layer-count and population-dependent noise recipes
│   ├── dust_ext.py         # DUSTQ: exact upstream estimator with configurable sigmas
│   ├── backprop_ext.py     # BP-STE and BP-SSTE training step implementations
│   ├── train.py            # Resumable, crash-safe training Runner and evaluation
│   ├── runner.py           # Smoke-testing harness and model builder
│   ├── queue.py            # Idempotent queue runner with retry markers
│   └── upstream.py         # Path management for vendored upstream code
├── scripts/                # Execution & evaluation tooling
│   ├── benchmark.py        # Throughput, memory, and 2-concurrent worker timing
│   ├── colab_t1.py         # Self-contained Google Colab L4 execution runner
│   ├── run_queue.py        # CLI for batch experiment execution
│   ├── make_configs.py     # Generates reproducible tier configs (e.g. Tier R)
│   └── check_data.py       # Validates FineWeb tokenized data integrity & disjointness
├── configs/                # Pre-generated YAML experiment configurations
├── tests/                  # Automated verification suite
│   ├── test_phase0.py      # Core unit tests (bitwise parity, sizes, quantizers)
│   ├── test_t1.py          # Exact resume, queue idempotency, benchmark smoke
│   └── test_cuda.py        # GPU-only single-step sanity tests
├── docs/                   # Scientific specifications & planning
│   ├── planning/           # Dust-Q Implementation Plan v3.1 (canonical)
│   ├── HYPOTHESES.md       # Pre-registered hypotheses and evaluation endpoints
│   ├── REPO_MAP.md         # Upstream architectural mapping and design notes
│   ├── PUBLICATION_STAGES.md # Gate criteria, paper structure, and venue roadmap
│   └── reports/            # Milestone verification reports (T0, T1, review fixes)
└── third_party/dust/       # Pinned upstream Q Labs Dust repository
```

---

## Quickstart

### 1. Installation

Create a virtual environment and install dependencies:

```bash
# Clone the repository
git clone https://github.com/your-org/dust-q.git
cd dust-q

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install PyTorch and testing tools
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or with CUDA support
pip install pytest pyyaml datasets tokenizers
```

### 2. Run the Test Suite

Verify all unit contracts and acceptance tests:

```bash
pytest tests -v
```

*Expected output: 27 passed, 4 skipped (CUDA tests skip automatically on CPU environments).*

### 3. Run a Smoke Step

Execute a minimal Dust-Q training step locally on CPU:

```python
from dustq import runner

# Run 3 steps of Dust on a quantized W1.58A4 model
losses = runner.run_dust(precision="w1.58a4", steps=3, population=256)
print("Step losses:", losses)
```

---

## Model Sizes & Supported Precisions

### Model Configurations

| Name | Alias | Layers | Hidden Dim ($d$) | Heads ($n_h$) | KV Heads ($n_{kv}$) | Sequence Length | Target Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tiny` | — | 2 | 64 | 8 | 8 | 128 | ~0.2M |
| `L2d128` | — | 2 | 128 | 2 | 2 | 2048 | ~2.1M |
| `L4d256` | **S** | 4 | 256 | 4 | 4 | 2048 | ~6.8M |
| `L8d512` | **M** | 8 | 512 | 8 | 8 | 2048 | ~37.5M |
| `L16d1024`| **L** | 16 | 1024 | 16 | 16 | 2048 | ~238M |

### Precision Formats

| Identifier | Weight Representation | Activation Bits | Quantization Scheme | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `fp` | bfloat16 / float32 | Unquantized | None | Full-precision baseline |
| `w8a8` | 8-bit Integer ($\pm 127$) | 8-bit | Per-token Absmax | Mild quantization |
| `w1.58a8` | Ternary $\{-1, 0, 1\}$ | 8-bit | Per-token Absmax | BitNet weights, 8-bit act |
| `w1.58a4` | Ternary $\{-1, 0, 1\}$ | 4-bit | Per-token Absmax / MSE | **Primary confirmatory cell** |
| `w1.58a2` | Ternary $\{-1, 0, 1\}$ | 2-bit | MSE-calibrated $\alpha_b$ | Ultra-low-bit exploratory |

---

## Experiments & Benchmarking

### Real-Scale GPU Benchmarking

Measure throughput (seconds/step) and peak VRAM across model sizes and population counts ($K \in \{256, 1024, 4096\}$):

```bash
python scripts/benchmark.py --device cuda --scale real --sizes S,M --populations 256,1024,4096 --precisions fp,w1.58a4
```

### Running the Experiment Queue

Execute experiment batches defined in `configs/`:

```bash
# Run all pending experiments
python scripts/run_queue.py --configs-dir configs --results-dir results

# Filter by specific condition (e.g. only 10M token runs)
python scripts/run_queue.py --only "10M"

# Retry failed runs
python scripts/run_queue.py --retry-failed
```

### Running on Google Colab (L4 GPU)

1. Package the repository:
   ```bash
   git archive -o dustq.zip HEAD
   ```
2. Upload `dustq.zip` to your Colab session.
3. Open a fresh notebook with an **L4 GPU** runtime.
4. Copy the entire contents of [`scripts/colab_t1.py`](scripts/colab_t1.py) into a single cell and run it. The script automatically validates code integrity, runs GPU smoke checks, benchmarks real-scale models, and formats the output table.

---

## Dataset Preparation

Data is tokenized using the FineWeb sample-10BT corpus with the pinned 4,096-token BPE tokenizer:

```bash
# Prepare 1M and 10M token training, validation, and test splits
python third_party/dust/prepare_data.py --tokens both --output data

# Verify dataset disjointness, sequence length (2,049 tokens), and SHA256 hashes
python scripts/check_data.py --data-dir data
```

---

## Research Documentation

Detailed documentation and research logs are tracked in `docs/`:

- [**Implementation Plan v3.1**](docs/planning/Dust-Q%20Implementation%20Plan%20v3.md): The canonical design plan, cost model, and gate definitions.
- [**Pre-registered Hypotheses**](docs/HYPOTHESES.md): Primary and secondary endpoints, paired-seed bootstrap methodology, and early exit criteria.
- [**Publication Stages & Venues**](docs/PUBLICATION_STAGES.md): Submission criteria, fallback paths (TMLR $\to$ Neural Networks $\to$ Neurocomputing), and target narrative.
- [**Upstream Repository Map**](docs/REPO_MAP.md): Detailed line-by-line mapping against upstream Q Labs `dust.py`.
- [**Open Questions & Tracking**](docs/OPEN_QUESTIONS.md): Active tracking of experimental and theoretical questions.

---

## License & Attribution

This project is licensed under the [MIT License](LICENSE).

Upstream Dust algorithm and implementation:
```bibtex
@misc{qlabs2025dust,
  title={DUST: Activation-Space Zeroth-Order Optimization for Transformers},
  author={Q Labs},
  year={2025},
  url={https://github.com/qlabs-eng/dust}
}
```
