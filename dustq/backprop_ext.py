"""Backprop training steps: BP (fp), BP-STE and later BP-SSTE (Plan v3.1 section 3.1).

Mirrors baselines/backprop.py exactly: bf16 autocast on CUDA only, non-finite guards, and
its *own* optimizer grouping — wte at lr 1000, everything else at the base lr, with no
special lr for value_embeds or the lambda scalars (unlike DUST.make_optimizer). Any change
to grouping must be symmetric across methods and recorded in the paper.
"""

from __future__ import annotations

import torch


def make_bp_optimizer(model, lr: float, momentum: float):
    groups = [
        {"params": [p for name, p in model.named_parameters() if name != "transformer.wte.weight"], "lr": lr},
        {"params": [model.transformer.wte.weight], "lr": 1000.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def train_step(model, optimizer, x, y) -> float:
    """One backprop step; returns the batch loss (baselines/backprop.py train_step)."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        loss = model(x, y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite backpropagation loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite backpropagation gradient for {name}")
    optimizer.step()
    return loss.item()
