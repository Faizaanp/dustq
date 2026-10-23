"""Build upstream GPT and swap in the quantized forward (Plan v3.1 section 4.2).

What is quantized (and where our hooks sit):
  - weight-quantized Linears: attn.c_q, attn.c_k, attn.c_v, attn.c_proj, mlp.c_fc, mlp.c_proj
    (swapped in place for quant.QLinear; lm_head, ve_gate and embeddings stay high-precision);
  - activation quantization: forward pre-hooks on the *inputs* of those same Linears:
    c_q/c_k/c_v share norm(x); attn.c_proj's input is the attention output o (the site Dust
    jitters as o@l); mlp.c_fc's input is norm(x); mlp.c_proj's input is the ReLU^2 output
    (unsigned grid). Everything else (q/k after rotary+norm, attention probabilities,
    ve_gate, embeddings, value embeddings, LM head, norms, lambdas) is left exact, which is
    what keeps upstream's local_errors and head_error valid (Plan v3.1 sections 2 and 4.2).

Hook ordering: upstream registers its o@l pre-hook inside DUST.__init__, so activation
pre-hooks must be registered AFTER the DUSTQ/DUST instance is constructed, giving the
order noise -> quantize. runner.run_dust guarantees this; tests 3-4 of T5 verify it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .dust_ext import DEFAULT_SIGMAS
from .quant import QLinear, sste_act_quantize, ste_act_quantize, swap_linear

WEIGHT_QUANT_SITES = ("attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
ACT_QUANT_SITES = WEIGHT_QUANT_SITES  # inputs of the weight-quantized Linears
_UNSIGNED_SITES = ("mlp.c_proj",)  # post-ReLU^2 input uses the unsigned grid


def set_exact_path(model: nn.Module, enabled: bool) -> None:
    """Enable QLinear's integer-code path (Dust runs only: BP-STE/SSTE keep fake-quant
    train-eval consistency by design; Plan v3.1 4.2)."""
    for module in model.modules():
        if isinstance(module, QLinear):
            module._exact_path = enabled


def site_sigma(name: str, n_layer: int) -> float:
    """Dust's per-site noise scale matched to our activation-quant sites [design]:
    c_q/c_k/c_v -> the local-internals sigma (their own linears' jitter site);
    attn.c_proj -> the o@l hub sigma (0.2 shallow half / 0.4 deep half, by layer);
    mlp.c_fc -> the hidden sigma; mlp.c_proj -> the writer sigma. Single source of
    truth: dustq.dust_ext.DEFAULT_SIGMAS."""
    if name.endswith("attn.c_proj"):
        layer = int(name.split(".h.")[1].split(".")[0])
        return DEFAULT_SIGMAS["hub_shallow"] if layer < n_layer // 2 else DEFAULT_SIGMAS["hub_deep"]
    if name.endswith(("attn.c_q", "attn.c_k", "attn.c_v")):
        return DEFAULT_SIGMAS["local"]
    if name.endswith("mlp.c_fc"):
        return DEFAULT_SIGMAS["hidden"]
    return DEFAULT_SIGMAS["writer"]


def parse_precision(precision: str) -> tuple[str, int, str]:
    """'fp', 'w8a8', 'w1.58a8', 'w1.58a4', 'w1.58a2'; run-id spellings ('w1p58a4') accepted."""
    aliases = {"w1p58a4": "w1.58a4", "w1p58a8": "w1.58a8", "w1p58a2": "w1.58a2"}
    key = aliases.get(precision.lower(), precision.lower()).replace("_", "").replace(".", "")
    table = {
        "fp": ("fp", 0, "absmax"),
        "w8a8": ("int8", 8, "absmax"),       # A8 may keep absmax (Plan v3.1 4.2)
        "w158a8": ("ternary", 8, "absmax"),
        "w158a4": ("ternary", 4, "mse_alpha"),  # low-bit activations use calibrated alpha_b
        "w158a2": ("ternary", 2, "mse_alpha"),
    }
    if key not in table:
        raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(table)} (dotted or run-id spelling)")
    return table[key]


def _site_matches(name: str, sites) -> str | None:
    for site in sites:
        if name.endswith("." + site) or name == site:
            return site
    return None


def swap_quantized_linears(model: nn.Module, mode: str) -> nn.Module:
    """Replace the in-scope Linears with QLinear, preserving names and parameter shapes."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _site_matches(name, WEIGHT_QUANT_SITES):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1], swap_linear(module, mode))
    return model


_ALPHA_TABLE_PATH = Path(__file__).resolve().parents[1] / "configs" / "alpha_calibrated.json"
_ALPHA_TABLE: dict | None = None


def calibrated_alpha(bits: int, unsigned: bool) -> float:
    """alpha_b calibrated once on a backprop fp checkpoint (Plan v3.1 4.2), frozen in
    configs/alpha_calibrated.json by scripts/calibrate_alpha.py; never learned."""
    global _ALPHA_TABLE
    if _ALPHA_TABLE is None:
        if not _ALPHA_TABLE_PATH.is_file():
            raise FileNotFoundError(
                "configs/alpha_calibrated.json is missing: run scripts/calibrate_alpha.py "
                "(needs a backprop fp checkpoint) before any A4/A2 training/probe run")
        _ALPHA_TABLE = json.loads(_ALPHA_TABLE_PATH.read_text())
    key = f"a{bits}{'u' if unsigned else 's'}"
    if key not in _ALPHA_TABLE:
        raise KeyError(f"no calibrated alpha for {key}; recalibrate")
    return float(_ALPHA_TABLE[key])


def register_act_hooks(model: nn.Module, bits: int, scheme: str = "absmax",
                       allow_lowbit_absmax: bool = False, ste_mode: str = "identity",
                       n_layer: int | None = None, alpha: float | None = None) -> list:
    """Attach activation-quantization pre-hooks to the in-scope Linears.

    Must be called AFTER DUSTQ/DUST construction on the Dust path (hook order: noise then
    quantize). Returns the hook handles so tests can remove them.

    ste_mode: "identity" = BP-STE (round/clamp straight-through); "smoothed" = BP-SSTE
    (Gaussian-smoothed quantizer derivative with the site's Dust sigma). The hook also
    stashes integer codes + per-token scale on the module for QLinear's exact
    integer-code path (consumed only under no_grad, i.e. by the Dust estimator).

    C5 guard: bits <= 4 with absmax is degenerate (Plan v3.1 4.2) and is refused outside
    smoke/benchmark mode; real A4/A2 cells wait for T5's MSE-calibrated alpha_b.
    """
    if scheme == "absmax" and bits <= 4 and not allow_lowbit_absmax:
        raise ValueError(
            f"{bits}-bit activations with absmax scales are degenerate (Plan v3.1 4.2); "
            "smoke/tests pass allow_lowbit_absmax=True; real A4/A2 cells use mse_alpha"
        )
    if ste_mode not in ("identity", "smoothed"):
        raise ValueError(f"unknown ste_mode {ste_mode!r}")
    n_layer = n_layer if n_layer is not None else model.config.n_layer
    handles = []
    for name, module in model.named_modules():
        site = _site_matches(name, ACT_QUANT_SITES)
        if site is None or not isinstance(module, nn.Linear):
            continue
        unsigned = site in _UNSIGNED_SITES
        sigma = site_sigma(name, n_layer)
        if scheme == "mse_alpha" and alpha is None:
            alpha = calibrated_alpha(bits, unsigned)  # frozen table (Plan v3.1 4.2)

        def make_hook(bits=bits, unsigned=unsigned, scheme=scheme, ste_mode=ste_mode,
                      sigma=sigma, alpha=alpha):
            def hook(module, args):
                x = args[0]
                if not x.is_floating_point():
                    return args
                # Stash codes + scale for QLinear's exact path (identical values to the
                # dequantized input: same formulas, deterministic).
                if unsigned:
                    lo, hi = 0, 2 ** bits - 1
                else:
                    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
                if scheme == "absmax":
                    scale = x.detach().float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / hi
                else:  # mse_alpha (calibrated alpha; guarded by the C5 check above)
                    scale = alpha * x.detach().float().abs().mean(dim=-1, keepdim=True)
                codes = torch.clamp(torch.round(x.detach().float() / scale), lo, hi)
                module._act_codes = codes
                module._act_scale = scale
                if ste_mode == "smoothed":
                    out = sste_act_quantize(x, bits, unsigned=unsigned, scheme=scheme,
                                            sigma=sigma, alpha=alpha)
                else:
                    out = ste_act_quantize(x, bits, unsigned=unsigned, scheme=scheme, alpha=alpha)
                return (out,)
            return hook

        handles.append(module.register_forward_pre_hook(make_hook()))
    return handles
