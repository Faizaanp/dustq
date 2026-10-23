"""Fake-quantization primitives (Plan v3.1 section 4.2): fp32 math, latent fp32 weights,
identity STE on round and clamp.

Phase 0 scope: correct shapes, correct quantizer semantics, NaN-free training steps. Two
T5 refinements are intentionally deferred and recorded as deviations in docs/reports/T0.md:
  - exact integer-code matmuls (here we fake-quantize in fp32 and use F.linear);
  - MSE-calibrated clip multipliers alpha_b for A4/A2 (here absmax scales at every bit width).
The Gaussian-smoothed STE (BP-SSTE) autograd function also arrives in T5 (test 7).
"""

from __future__ import annotations

import math

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
        if self.weight.dtype != torch.float32:
            raise ValueError("QLinear latent weights must be fp32 (bf16-cast latents flip "
                             "codes at rounding thresholds — arXiv:2610.07853, OPEN_QUESTIONS 27)")
        # Code cache (T5, budget-critical): codes are recomputed only when the latent
        # weight is mutated in place (optimizer.step() bumps tensor._version). The old
        # per-forward recompute cost the measured 2.5-3.2x on every quantized run.
        self._codes = None
        self._gamma = None
        self._codes_version = -1
        # Activation-code stash set by the act-quant pre-hook (DUST exact path):
        self._act_codes = None
        self._act_scale = None
        # Exact integer-code path is DUST-only (set by qmodel.set_exact_path): BP-STE
        # and BP-SSTE keep the same fake-quant forward in train and eval by design.
        self._exact_path = False

    def _codes_and_gamma(self):
        w = self.weight
        if self._codes is None or self._codes_version != w._version:
            with torch.no_grad():
                if self.quant_mode == "ternary":
                    gamma = w.abs().mean()
                    codes = torch.clamp(torch.round(w / (gamma + 1e-5)), -1, 1)
                else:
                    gamma = w.abs().max().clamp(min=1e-5) / 127.0
                    codes = torch.clamp(torch.round(w / gamma), -128, 127)
            self._codes, self._gamma = codes, gamma
            self._codes_version = w._version
        return self._codes, self._gamma

    def effective_weight(self) -> torch.Tensor:
        """Forward value: gamma * codes exactly; backward: identity STE through the
        latent weight (codes/gamma are cached constants detached from the graph)."""
        codes, gamma = self._codes_and_gamma()
        w = self.weight
        return w + (gamma * codes - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes, gamma = self._codes_and_gamma()
        # Exact integer-code path (T5/4.2): used when no backward is needed (DUST's
        # forward-only estimator). Integer codes multiply exactly in fp32 accumulation
        # (|term| <= 127*127*4096 < 2^24), rescaled once in fp32 — no bf16 rounding can
        # move grid points. Autograd runs (BP-STE/BP-SSTE) take the fake-quant path so
        # the weight STE flows.
        if self._exact_path and not torch.is_grad_enabled() and self._act_codes is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                acc = self._act_codes.to(torch.float32) @ codes.to(torch.float32).transpose(0, 1)
                y = acc * self._act_scale.to(torch.float32) * gamma.to(torch.float32)
            if self.bias is not None:
                y = y + self.bias
            self._act_codes = None
            self._act_scale = None
            return y
        self._act_codes = None
        self._act_scale = None
        return F.linear(x, self.effective_weight(), self.bias)


class _SSTEActQuant(torch.autograd.Function):
    """BP-SSTE activation quantizer (Plan v3.1 3.1/4.2): forward = the hard-quantized
    value (identical to STE); backward = the derivative of the Gaussian-smoothed
    quantizer Q_sigma(x) = E[q(x + sigma*e)]:

        dQ/dx = sum over INTERIOR grid boundaries b of phi((x - s*b)/sigma)/sigma

    Each interior boundary contributes a Gaussian bump in activation units; boundaries
    beyond the code range are excluded (crossing them is clamped away); the sum is
    truncated to the nearest +/-3 boundaries (negligible beyond). Derived from
    d/du E[round(u + sigma_u*e)] = sum_b phi((u - b)/sigma_u)/sigma_u with u = x/s and
    sigma_u = sigma/s; verified against Monte-Carlo finite differences in
    test_sste_backward_matches_mc."""

    @staticmethod
    def forward(ctx, x, scale, lo, hi, sigma):
        codes = torch.clamp(torch.round(x / scale), lo, hi)
        ctx.save_for_backward(x, scale)
        ctx.lo, ctx.hi, ctx.sigma = lo, hi, sigma
        return codes * scale

    @staticmethod
    def backward(ctx, grad_out):
        x, scale = ctx.saved_tensors
        lo, hi, sigma = ctx.lo, ctx.hi, ctx.sigma
        u = x / scale
        j0 = torch.floor(u + 0.5)  # nearest code index
        gx = torch.zeros_like(x)
        for d in range(-3, 4):
            k = j0 + d                       # boundary sits between codes k-1 and k
            b = k - 0.5
            valid = (k >= lo + 1) & (k <= hi) & ((u - b).abs() <= 4 * (sigma / scale))
            bump = torch.exp(-0.5 * ((x - scale * b) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
            gx = gx + torch.where(valid, bump, torch.zeros_like(bump))
        return grad_out * gx, None, None, None, None


def sste_act_quantize(x: torch.Tensor, bits: int, unsigned: bool = False,
                      scheme: str = "absmax", sigma: float = 0.2,
                      alpha: float | None = None) -> torch.Tensor:
    """BP-SSTE variant of ste_act_quantize: same forward value, smoothed backward."""
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
    return _SSTEActQuant.apply(x, scale, lo, hi, sigma)


def swap_linear(module: nn.Linear, mode: str) -> QLinear:
    """Replace a Linear with a QLinear on the same DEVICE (B4: a CPU-built QLinear would
    crash every CUDA run). The latent is normalized to fp32 regardless of the source
    dtype: bf16 latents flip codes at rounding thresholds (arXiv:2610.07853; OQ-27) —
    enforced by an assertion in QLinear.__init__."""
    q = QLinear(module.in_features, module.out_features, mode=mode,
                bias=module.bias is not None,
                device=module.weight.device, dtype=torch.float32)
    q.weight.data.copy_(module.weight.data)  # copy_ casts bf16 -> fp32 if needed
    if module.bias is not None:
        q.bias.data.copy_(module.bias.data)
    return q
