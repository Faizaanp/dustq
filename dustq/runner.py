"""One entry point for all methods (Plan v3.1 section 4.4). Phase 0 scope: CPU smoke tests
and the DUSTQ-vs-DUST bitwise harness. Config files, queueing, resume, Drive paths and
checkpointing arrive in T1.

Hook ordering on the Dust path is load-bearing: the model is built (QLinear swap), then
DUSTQ is constructed (upstream registers its o@l pre-hook inside __init__), and only then
are the activation-quantization pre-hooks attached — giving noise -> quantize.
"""

from __future__ import annotations

import argparse

import torch

from .upstream import ensure_upstream_on_path

ensure_upstream_on_path()

from dust import GPTConfig, make_model  # noqa: E402
from dust import make_optimizer as dust_make_optimizer  # noqa: E402
from dust import DUST  # noqa: E402

from . import qmodel  # noqa: E402
from .backprop_ext import make_bp_optimizer, train_step  # noqa: E402
from .dust_ext import DUSTQ  # noqa: E402
from .recipes import recipe  # noqa: E402

# Tiny CPU acceptance config (Plan v3.1 section 7, Phase 0 acceptance).
TINY = dict(n_layer=2, n_embd=64, n_head=8, sequence_len=128, vocab_size=4096)


def tiny_batch(device, seed: int = 1234, sequences: int = 8, length: int = 128, vocab: int = 4096):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (sequences, length + 1), generator=generator).to(device)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def build_model(cfg: dict | None = None, precision: str = "fp", device: str | device = "cpu", seed: int = 0):
    config = GPTConfig(**(cfg or TINY))
    model = make_model(config, torch.device(device), seed)
    if precision != "fp":
        mode, _, _ = qmodel.parse_precision(precision)
        qmodel.swap_quantized_linears(model, mode)
    return model


def run_dust(precision: str = "fp", steps: int = 3, population: int = 256, tokens: str = "1M",
             seed: int = 0, device: str | device = "cpu", cfg: dict | None = None,
             sigmas: dict | None = None) -> list[float]:
    model = build_model(cfg, precision, device, seed).requires_grad_(False)
    settings = recipe(tokens, population, model.config.n_layer)
    estimator = DUSTQ(model, settings, seed=seed, sigmas=sigmas)
    if precision != "fp":
        _, act_bits, act_scheme = qmodel.parse_precision(precision)
        qmodel.register_act_hooks(model, act_bits, act_scheme)  # AFTER DUSTQ: noise -> quantize
    optimizer = dust_make_optimizer(model, settings)
    x, y = tiny_batch(torch.device(device) if isinstance(device, str) else device)
    losses = []
    for _ in range(steps):
        losses.append(estimator.step(x, y, optimizer))
    estimator.close()
    return losses


def run_backprop(precision: str = "w1.58a4", steps: int = 3, seed: int = 0,
                 device: str | device = "cpu", cfg: dict | None = None,
                 lr: float = 1e-3, momentum: float = 0.9) -> list[float]:
    # lr default is a gentle smoke value (deviation recorded in T0 report): the acceptance
    # criterion is NaN-free steps through the quantized forward + STE, not tuned training.
    model = build_model(cfg, precision, device, seed).requires_grad_(True)
    if precision != "fp":
        _, act_bits, act_scheme = qmodel.parse_precision(precision)
        qmodel.register_act_hooks(model, act_bits, act_scheme)
    optimizer = make_bp_optimizer(model, lr, momentum)
    x, y = tiny_batch(torch.device(device) if isinstance(device, str) else device)
    losses = []
    for _ in range(steps):
        losses.append(train_step(model, optimizer, x, y))
    return losses


def main():
    parser = argparse.ArgumentParser(description="Phase 0 CPU smoke runner")
    parser.add_argument("--method", choices=["dust", "backprop"], default="dust")
    parser.add_argument("--precision", default="fp",
                        help="fp | w8a8 | w1.58a8 | w1.58a4 | w1.58a2")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--tokens", default="1M")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = run_dust if args.method == "dust" else run_backprop
    losses = run(precision=args.precision, steps=args.steps, population=args.population,
                 tokens=args.tokens, seed=args.seed)
    print({ "method": args.method, "precision": args.precision, "losses": losses })


if __name__ == "__main__":
    main()
