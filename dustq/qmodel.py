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

import torch
from torch import nn

from .quant import QLinear, ste_act_quantize, swap_linear

WEIGHT_QUANT_SITES = ("attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
ACT_QUANT_SITES = WEIGHT_QUANT_SITES  # inputs of the weight-quantized Linears
_UNSIGNED_SITES = ("mlp.c_proj",)  # post-ReLU^2 input uses the unsigned grid


def parse_precision(precision: str) -> tuple[str, int, str]:
    """'fp', 'w8a8', 'w1.58a8', 'w1.58a4', 'w1.58a2'; run-id spellings ('w1p58a4') accepted."""
    aliases = {"w1p58a4": "w1.58a4", "w1p58a8": "w1.58a8", "w1p58a2": "w1.58a2"}
    key = aliases.get(precision.lower(), precision.lower()).replace("_", "").replace(".", "")
    table = {
        "fp": ("fp", 0, "absmax"),
        "w8a8": ("int8", 8, "absmax"),
        "w158a8": ("ternary", 8, "absmax"),
        "w158a4": ("ternary", 4, "absmax"),
        "w158a2": ("ternary", 2, "absmax"),
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


def register_act_hooks(model: nn.Module, bits: int, scheme: str = "absmax",
                       allow_lowbit_absmax: bool = False) -> list:
    """Attach activation-quantization pre-hooks to the in-scope Linears.

    Must be called AFTER DUSTQ/DUST construction on the Dust path (hook order: noise then
    quantize). Returns the hook handles so tests can remove them.

    C5 guard: bits <= 4 with absmax is degenerate (Plan v3.1 4.2) and is refused outside
    smoke/benchmark mode; real A4/A2 cells wait for T5's MSE-calibrated alpha_b.
    """
    if scheme == "absmax" and bits <= 4 and not allow_lowbit_absmax:
        raise ValueError(
            f"{bits}-bit activations with absmax scales are degenerate (Plan v3.1 4.2); "
            "smoke/tests pass allow_lowbit_absmax=True; real A4/A2 cells wait for T5 alpha_b"
        )
    handles = []
    for name, module in model.named_modules():
        site = _site_matches(name, ACT_QUANT_SITES)
        if site is None or not isinstance(module, nn.Linear):
            continue
        unsigned = site in _UNSIGNED_SITES

        def make_hook(bits=bits, unsigned=unsigned, scheme=scheme):
            def hook(_module, args):
                x = args[0]
                if not x.is_floating_point():
                    return args
                return (ste_act_quantize(x, bits, unsigned=unsigned, scheme=scheme),)
            return hook

        handles.append(module.register_forward_pre_hook(make_hook()))
    return handles
