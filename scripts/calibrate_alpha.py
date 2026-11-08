"""T5 alpha_b calibration (Plan v3.1 4.2): choose the per-token clip multiplier alpha_b
once, by minimising quantization MSE on activations captured from a backprop fp
checkpoint; frozen into configs/alpha_calibrated.json; never learned.

Usage:
  python scripts/calibrate_alpha.py --checkpoint results_local/L4d256_1M_fp_backprop_K0_s0/best.pt \
      [--batches 8] [--grid 0.4 0.45 ... 3.2] [--out configs/alpha_calibrated.json]

Method: load the checkpoint, run forwards on real training batches with spy pre-hooks at
the six activation-quant sites (raw inputs, pre-quantization), pool the site activations
(signed sites and the unsigned ReLU^2 site separately), then for each bit width in
{8, 4, 2} grid-search alpha minimizing  mean((dequant(quant_alpha(x)) - x)^2)  with
scale = alpha * mean|x| per token (the production formula). Provenance is written into
the JSON (checkpoint, data, grid, mse, chosen alpha).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dustq import qmodel
from dustq.config import SIZES
from dustq.train import evaluate, load_best
from dustq.upstream import ensure_upstream_on_path

ensure_upstream_on_path()

from dust import load_sequences  # noqa: E402


def gather_site_activations(checkpoint: Path, data_dir: str, batches: int, device):
    _, state = load_best(checkpoint, device)
    from dustq.runner import build_model
    model = build_model(SIZES["L4d256"], "fp", device, seed=0)
    model.load_state_dict(state)
    model.eval()
    captured = {name: [] for name in
                ("attn.c_q", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")}

    def spy(name):
        def hook(_module, args):
            captured[name].append(args[0].detach().float().flatten())
        return hook

    for name, module in model.named_modules():
        short = next((s for s in captured if name.endswith(s)), None)
        if short and isinstance(module, torch.nn.Linear):
            module.register_forward_pre_hook(spy(short))

    train = load_sequences(Path(data_dir) / "train_1M.pt", 61 * 8)
    rows = train[:batches * 8]
    with torch.no_grad():
        for i in range(batches):
            chunk = rows[i * 8:(i + 1) * 8].to(device)
            x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
            model(x, y)
    return {k: torch.cat(v) for k, v in captured.items() if v}


def quant_mse(x: torch.Tensor, bits: int, unsigned: bool, alpha: float) -> float:
    if unsigned:
        lo, hi = 0, 2 ** bits - 1
    else:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    # clamp: a token whose activations are all zero (padding, dead ReLU^2) would give
    # scale = 0 and then 0/0 = NaN, poisoning the whole calibration search. The production
    # quantizers in dustq/quant.py clamp at the same floor, so matching it here keeps the
    # calibration objective and the runtime quantizer consistent.
    scale = (alpha * x.abs().mean(dim=-1, keepdim=True)).clamp(min=1e-5)
    codes = torch.clamp(torch.round(x / scale), lo, hi)
    err = codes * scale - x
    return (err ** 2).mean().item()


def _calibrate(x: torch.Tensor, bits: int, unsigned: bool, coarse_n: int = 32) -> tuple[float, float, bool]:
    """Two-stage search over log(alpha) in [-2, 2.2] (0.01 x to 158 x mean|x|; ~4 orders):
    coarse grid, then refine within the winning cell. 2026-10-31: the first calibration
    used a fixed [0.40, 3.20] linear grid and 3/6 optima pinned at the boundaries
    (a4u mse 0.177, a2u mse 0.665 — near-all-zero codes). Returns (alpha, mse, at_bounds)."""
    lo_exp, hi_exp = -2.0, 2.2
    coarse = [lo_exp + (hi_exp - lo_exp) * i / (coarse_n - 1) for i in range(coarse_n)]
    vals = [quant_mse(x, bits, unsigned, 10 ** e) for e in coarse]
    i = min(range(coarse_n), key=vals.__getitem__)
    step = (hi_exp - lo_exp) / (coarse_n - 1)
    rlo = max(coarse[max(i - 1, 0)], lo_exp)
    rhi = min(coarse[min(i + 1, coarse_n - 1)], hi_exp)
    fine = [rlo + (rhi - rlo) * j / 20 for j in range(21)]
    fvals = [quant_mse(x, bits, unsigned, 10 ** e) for e in fine]
    j = min(range(21), key=fvals.__getitem__)
    best_exp = fine[j]
    at_bounds = best_exp <= lo_exp + step / 2 or best_exp >= hi_exp - step / 2
    return 10 ** best_exp, fvals[j], at_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batches", type=int, default=8, help="8x2048-token batches to pool")
    parser.add_argument("--grid", nargs="+", type=float,
                        default=[round(0.4 + 0.05 * i, 2) for i in range(57)])  # 0.40..3.20
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 4, 2])
    parser.add_argument("--out", type=Path, default=Path("configs/alpha_calibrated.json"))
    args = parser.parse_args()

    device = torch.device("cpu")
    acts = gather_site_activations(args.checkpoint, args.data_dir, args.batches, device)
    # c_q/c_k/c_v share one input; pool signed sites (norm(x), o@l) and the unsigned
    # ReLU^2 site separately — the production scale is per (bits, signedness).
    signed = torch.cat([acts["attn.c_q"], acts["attn.c_proj"], acts["mlp.c_fc"]])
    unsigned = acts["mlp.c_proj"]
    print(f"pooled: signed {signed.numel():,} values | unsigned {unsigned.numel():,} values")

    table, report = {}, {}
    for bits in args.bits:
        for tag, x in (("s", signed), ("u", unsigned)):
            alpha, mse, at_bounds = _calibrate(x, bits, tag == "u")
            key = f"a{bits}{tag}"
            table[key] = alpha
            report[key] = {"alpha": alpha, "mse": mse, "at_grid_bounds": at_bounds}
            flag = "  <-- AT GRID BOUNDS, widen further" if at_bounds else ""
            print(f"{key}: alpha={alpha:.4f} mse={mse:.6f}{flag}")

    payload = {
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "data": f"{args.data_dir}/train_1M.pt (first {args.batches} batches)",
            "grid": "two-stage log-alpha in [-2.0, 2.2] (0.01x..158x mean|x|), 32pt coarse + 21pt refine",
            "sites": "signed: c_q/c_k/c_v + o@l + c_fc inputs; unsigned: mlp.c_proj input",
            "rule": "scale = alpha_b * mean|x| per token; argmin MSE over the grid",
        },
        "alpha": table,
        "detail": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
