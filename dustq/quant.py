"""Fake-quantization primitives (Plan v3.1 section 4.2): fp32 math, latent fp32 weights,
identity STE on round and clamp.

Phase 0 scope: correct shapes, correct quantizer semantics, NaN-free training steps. Two
T5 refinements are intentionally deferred and recorded as deviations in docs/reports/T0.md:
  - exact integer-code matmuls (here we fake-quantize in fp32 and use F.linear);
  - MSE-calibrated clip multipliers alpha_b for A4/A2 (here absmax scales at every bit width).
The Gaussian-smoothed STE (BP-SSTE) autograd function also arrives in T5 (test 7).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def ste_quantize(x: torch.Tensor, scale, lo: int, hi: int) -> torch.Tensor:
    """clamp(round(x / scale), lo, hi) * scale with identity backward through round and clamp.

    torch.round has a zero-gradient backward in PyTorch, so the STE detach pattern must be
    applied to round itself (not just to clamp), or the whole path's gradient dies.
    """
    y = x / scale
    y = y + (torch.round(y) - y).detach()          # forward: round(y); backward: identity
    y = y + (torch.clamp(y, lo, hi) - y).detach()  # forward: clamp(y); backward: identity
    return y * scale


def ste_act_quantize(x: torch.Tensor, bits: int, unsigned: bool = False,
                     scheme: str = "absmax", alpha: float | None = None) -> torch.Tensor:
    """Per-token activation quantizer over the last (hidden) dimension.

    Quantizer math runs in fp32 (Plan v3.1 4.2) regardless of the input dtype; the output
    is fp32 (autocast casts per-op on CUDA; DUST captures args[0].float() either way).
    Residual caveat: under autocast the matmul still casts the dequantized values to bf16,
    so grid distortion persists until T5's integer-code matmul (spec, C1).

    absmax (Phase 0): scale = max|x| / hi. mse_alpha (T5): scale = alpha * mean|x|.
    Unsigned grid is used for the ReLU^2 output feeding mlp.c_proj (Plan v3.1 4.2).
    """
    x = x.float()
    if unsigned:
        lo, hi = 0, 2 ** bits - 1
    else:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    if scheme == "absmax":
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / hi
    elif scheme == "mse_alpha":
        if alpha is None:
            raise ValueError("mse_alpha scheme requires an explicit alpha (calibrated in T5)")
        scale = alpha * x.detach().abs().mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown activation scheme {scheme!r}")
    return ste_quantize(x, scale, lo, hi)


class QLinear(nn.Linear):
    """Linear with a fake-quantized weight and identity STE; the latent fp32 weight is the
    parameter that optimizers and DUST's estimator both see (grads land on the latent).

    Subclasses nn.Linear so upstream DUST's isinstance-based hook capture and the
    optimizer parameter groups see exactly the same module names and parameters as at fp.
    """

    def __init__(self, in_features: int, out_features: int, mode: str = "ternary",
                 bias: bool = False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if mode not in ("ternary", "int8"):
            raise ValueError(f"unknown weight-quant mode {mode!r}")
        self.quant_mode = mode

    def effective_weight(self) -> torch.Tensor:
        w = self.weight
        if self.quant_mode == "ternary":
            # BitNet b1.58-style: gamma = mean|W|; codes in {-1, 0, 1}; effective = gamma * codes.
            gamma = w.abs().mean()
            codes = torch.clamp(torch.round(w / (gamma + 1e-5)), -1, 1)
        else:
            scale = w.abs().max().clamp(min=1e-5) / 127.0
            codes = torch.clamp(torch.round(w / scale), -128, 127)
            gamma = scale
        # Forward value: gamma * codes exactly; backward: identity (straight-through).
        return w + (gamma * codes - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


def swap_linear(module: nn.Linear, mode: str) -> QLinear:
    """Replace a Linear with a QLinear on the same device and dtype (B4: the swap must be
    transparent to a CUDA-placed model; a CPU-built QLinear would crash every CUDA run)."""
    q = QLinear(module.in_features, module.out_features, mode=mode,
                bias=module.bias is not None,
                device=module.weight.device, dtype=module.weight.dtype)
    q.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        q.bias.data.copy_(module.bias.data)
    return q
