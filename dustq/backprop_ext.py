"""Backprop training steps: BP (fp), BP-STE and later BP-SSTE (Plan v3.1 section 3.1).

Mirrors baselines/backprop.py exactly: bf16 autocast on CUDA only, non-finite guards, and
its *own* optimizer grouping — wte at lr 1000, everything else at the base lr, with no
special lr for value_embeds or the lambda scalars (unlike DUST.make_optimizer). Any change
to grouping must be symmetric across methods and recorded in the paper.

The 37.7M model's backprop graph at 16k tokens exceeds an L4's 22 GB: upstream's
attention passes an explicit dense mask to SDPA, which blocks the fused backends and
forces math-backend score materialization ((B,H,2048,2048) per layer plus gradients).
train_step therefore retries on CUDA OOM with 2, then 4, gradient-accumulation chunks —
numerically equivalent to the full-batch step (grads scaled by exact token counts).
Upstream never hit this because it ran on larger GPUs (REPO_MAP, memory note).
"""

from __future__ import annotations

import torch


def make_bp_optimizer(model, lr: float, momentum: float):
    groups = [
        {"params": [p for name, p in model.named_parameters() if name != "transformer.wte.weight"], "lr": lr},
        {"params": [model.transformer.wte.weight], "lr": 1000.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def _bp_step(model, optimizer, x, y, microbatches: int) -> float:
    """One backprop step over the batch, optionally split into equal chunks with
    gradient accumulation (token-count-weighted; equivalent to the full batch)."""
    optimizer.zero_grad(set_to_none=True)
    n = microbatches
    bs = x.shape[0]
    if bs % n:
        raise ValueError(f"batch {bs} not divisible by {n} microbatches")
    tokens_total = float((y != -1).sum())
    total_loss, tokens_seen = 0.0, 0.0
    for i in range(n):
        xb, yb = x[i * bs // n:(i + 1) * bs // n], y[i * bs // n:(i + 1) * bs // n]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            loss = model(xb, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite backpropagation loss")
        w = float((yb != -1).sum())
        (loss * (w / tokens_total)).backward()
        total_loss += loss.item() * w
        tokens_seen += w
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite backpropagation gradient for {name}")
    optimizer.step()
    return total_loss / tokens_seen


def train_step(model, optimizer, x, y) -> float:
    """One backprop step; returns the batch loss (baselines/backprop.py train_step),
    with automatic micro-batching on CUDA OOM. The chunk count that worked is cached on
    the model (`model._bp_microbatches`), so later steps start at the cached size instead
    of re-paying a wasted full-batch OOM every step (found in review: each retry burned a
    ~20 GB partial graph; that matters at 610-step budgets)."""
    if not x.is_cuda:
        return _bp_step(model, optimizer, x, y, 1)
    n = getattr(model, "_bp_microbatches", 1)
    while True:
        try:
            loss = _bp_step(model, optimizer, x, y, n)
            if n > 1:
                model._bp_microbatches = n
            return loss
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if n >= 8:
                raise
            n *= 2
