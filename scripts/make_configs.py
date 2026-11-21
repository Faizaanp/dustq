"""Emit the run configs for a tier into configs/ (one YAML per run, named by run_id).

Tiers (Plan v3.1 sections 6.2, 6.5, 7):
  R:    Tier R reproduction (M/1M/P0 + S/10M/P0, 3 seeds)
  tune: T7 hyperparameter tuning on validation loss (S/1M, seeds 100+)
  B-S:  Tier B Small (S/1M, P0-P4 x {BPx, K256, K1k}, 3 seeds)
  B-M:  Tier B Medium (M/1M, P2, P3 x {BPx, K256, K1k}, 3 seeds, K1k at 1 seed)
  A:    Tier A scaling (S/10M, P0, P3 x {BPx, K256, K1k})

Usage: python scripts/make_configs.py --tier {R,tune,B-S,B-M,A} [--seeds 0,1,2] [--out configs] [--tuned-hparams results/tune_best_hparams.json]
"""

import argparse
import json
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dustq.upstream import ensure_upstream_on_path  # noqa: E402

ensure_upstream_on_path()

import dust  # noqa: E402
from dustq.config import BP_DEFAULTS, RunConfig  # noqa: E402

# Optimal noise scales for W1.58A4 (P3) empirically calibrated from Probe 5 (Plan v3.1 section 6.5)
P3_SIGMAS = {
    "hidden": 0.420089,
    "hub_deep": 0.110207,
    "hub_shallow": 0.084098,
    "local": 0.420038,
    "writer": 0.971743,
}


def _lookup_tuned(tuned: dict | None, prec: str, method: str, ste_mode: str | None,
                  pop: int, tokens: str = "1M") -> tuple[float | None, float | None]:
    """Look up tuned hyperparameters from T7 results if available, else return (None, None)."""
    if not tuned:
        return None, None
    if method == "backprop":
        mode = ste_mode or "identity"
        cell = f"{prec}/backprop/{mode}"
    else:
        cell = f"{prec}/dust/K{pop}"
    info = tuned.get(cell)
    if not info:
        return None, None
    best_lr = info.get("best_lr")
    best_mom = info.get("best_momentum")
    if best_lr is None or best_mom is None:
        return None, None
    if tokens == "1M":
        return float(best_lr), float(best_mom)

    # For 10M, inherit S/1M multiplier on top of 10M default (Plan v3.1 section 6.5)
    if method == "backprop":
        def_1m = BP_DEFAULTS["1M"]["lr"]
        def_10m = BP_DEFAULTS["10M"]["lr"]
        mult = best_lr / def_1m
        return round(def_10m * mult, 6), float(best_mom)
    else:
        def_1m = dust.DEFAULT_CONFIGS["1M"]["optimizer"][pop]["lr"]
        def_10m = dust.DEFAULT_CONFIGS["10M"]["optimizer"][pop]["lr"]
        mult = best_lr / def_1m
        return round(def_10m * mult, 6), float(best_mom)


def tier_r(seeds: list[int], tuned: dict | None = None) -> list[RunConfig]:
    cells = []
    for seed in seeds:
        cells.append(RunConfig(task="R", size="L8d512", tokens="1M", precision="fp",
                               method="backprop", seed=seed))
        for k in (256, 1024, 4096):
            cells.append(RunConfig(task="R", size="L8d512", tokens="1M", precision="fp",
                                   method="dust", population=k, seed=seed))
        cells.append(RunConfig(task="R", size="L4d256", tokens="10M", precision="fp",
                               method="backprop", seed=seed))
        cells.append(RunConfig(task="R", size="L4d256", tokens="10M", precision="fp",
                               method="dust", population=256, seed=seed))
    return cells


def tier_tune(seeds: list[int], tuned: dict | None = None) -> list[RunConfig]:
    """T7 tuning protocol (Plan v3.1 section 6.5): validation-only selection.

    Seeds are strictly 100+ (never count as reported experimental seeds).
    - Dust LR: 3-point grid {0.6, 1.0, 1.5}x default for P3 and P0; 2-point {1.0, 1.5}x for P1, P2, P4.
    - Backprop: 5-point LR grid x 2 momenta per precision.
    """
    cells = []
    seed = seeds[0] if seeds else 100
    if seed < 100:
        seed = 100

    # 1. Dust LR grid
    dust_mults_p3_p0 = [0.6, 1.0, 1.5]
    dust_mults_others = [1.0, 1.5]
    precisions = [("fp", dust_mults_p3_p0),
                  ("w8a8", dust_mults_others),
                  ("w1.58a8", dust_mults_others),
                  ("w1.58a4", dust_mults_p3_p0),
                  ("w1.58a2", dust_mults_others)]

    for prec, mults in precisions:
        sigmas = P3_SIGMAS if prec == "w1.58a4" else None
        for k in (256, 1024):
            base_opt = dust.DEFAULT_CONFIGS["1M"]["optimizer"][k]
            def_lr = base_opt["lr"]
            def_mom = base_opt["momentum"]
            for m in mults:
                lr = round(def_lr * m, 6)
                cells.append(RunConfig(
                    task="tune", size="L4d256", tokens="1M", precision=prec,
                    method="dust", population=k, seed=seed, lr=lr, momentum=def_mom,
                    sigmas=sigmas, dust_forward="fake",
                ))

    # 2. Backprop LR x Momentum grid
    bp_def_lr = BP_DEFAULTS["1M"]["lr"]  # 0.175
    bp_lrs = [round(bp_def_lr * f, 6) for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
    bp_moms = [0.90, 0.98]
    all_precs = ["fp", "w8a8", "w1.58a8", "w1.58a4", "w1.58a2"]

    for prec in all_precs:
        sigmas = P3_SIGMAS if prec == "w1.58a4" else None
        for lr in bp_lrs:
            for mom in bp_moms:
                if prec == "fp":
                    cells.append(RunConfig(
                        task="tune", size="L4d256", tokens="1M", precision=prec,
                        method="backprop", seed=seed, lr=lr, momentum=mom,
                    ))
                else:
                    # BP-STE
                    cells.append(RunConfig(
                        task="tune", size="L4d256", tokens="1M", precision=prec,
                        method="backprop", ste_mode="identity", seed=seed, lr=lr, momentum=mom,
                    ))
                    # BP-SSTE
                    cells.append(RunConfig(
                        task="tune", size="L4d256", tokens="1M", precision=prec,
                        method="backprop", ste_mode="smoothed", sigmas=sigmas, seed=seed, lr=lr, momentum=mom,
                    ))

    return cells


def tier_b_s(seeds: list[int], tuned: dict | None = None) -> list[RunConfig]:
    """Tier B Small (Plan v3.1 section 6.2): S/1M across P0-P4 x {BPx, K256, K1k}.

    BPx = BP for P0; BP-STE + BP-SSTE for P1-P4.
    K=4k is fp only, 1 seed (seed 0).
    Uses tuned hyperparameters from T7 if provided.
    """
    cells = []
    precs = ["fp", "w8a8", "w1.58a8", "w1.58a4", "w1.58a2"]
    for seed in seeds:
        for prec in precs:
            sigmas = P3_SIGMAS if prec == "w1.58a4" else None
            # Backprop estimators
            if prec == "fp":
                lr, mom = _lookup_tuned(tuned, prec, "backprop", "identity", 0, "1M")
                cells.append(RunConfig(task="B", size="L4d256", tokens="1M", precision=prec,
                                       method="backprop", seed=seed, lr=lr, momentum=mom))
            else:
                lr_ste, mom_ste = _lookup_tuned(tuned, prec, "backprop", "identity", 0, "1M")
                cells.append(RunConfig(task="B", size="L4d256", tokens="1M", precision=prec,
                                       method="backprop", ste_mode="identity", seed=seed,
                                       lr=lr_ste, momentum=mom_ste))
                lr_sste, mom_sste = _lookup_tuned(tuned, prec, "backprop", "smoothed", 0, "1M")
                cells.append(RunConfig(task="B", size="L4d256", tokens="1M", precision=prec,
                                       method="backprop", ste_mode="smoothed", sigmas=sigmas, seed=seed,
                                       lr=lr_sste, momentum=mom_sste))

            # Dust estimators
            for k in (256, 1024):
                lr_k, mom_k = _lookup_tuned(tuned, prec, "dust", None, k, "1M")
                cells.append(RunConfig(task="B", size="L4d256", tokens="1M", precision=prec,
                                       method="dust", population=k, seed=seed, lr=lr_k, momentum=mom_k,
                                       sigmas=sigmas, dust_forward="fake"))

    # K=4k fp only, 1 seed (seed 0) [re-cut]
    lr_4k, _ = _lookup_tuned(tuned, "fp", "dust", None, 1024, "1M")  # inherits K1k LR; keeps K4k default momentum (0.98)
    cells.append(RunConfig(task="B", size="L4d256", tokens="1M", precision="fp",
                           method="dust", population=4096, seed=0, lr=lr_4k, momentum=None,
                           dust_forward="fake"))
    return cells


def tier_b_m(seeds: list[int], tuned: dict | None = None) -> list[RunConfig]:
    """Tier B Medium (Plan v3.1 section 6.2): M/1M across P2, P3 x {BPx, K256, K1k}.

    K1k at 1 seed (seed 0) [re-cut]. P0 reused from Tier R.
    Inherits S/1M tuned hyperparameters.
    """
    cells = []
    precs = ["w1.58a8", "w1.58a4"]
    for seed in seeds:
        for prec in precs:
            sigmas = P3_SIGMAS if prec == "w1.58a4" else None
            # Backprop estimators
            lr_ste, mom_ste = _lookup_tuned(tuned, prec, "backprop", "identity", 0, "1M")
            cells.append(RunConfig(task="B", size="L8d512", tokens="1M", precision=prec,
                                   method="backprop", ste_mode="identity", seed=seed,
                                   lr=lr_ste, momentum=mom_ste))
            lr_sste, mom_sste = _lookup_tuned(tuned, prec, "backprop", "smoothed", 0, "1M")
            cells.append(RunConfig(task="B", size="L8d512", tokens="1M", precision=prec,
                                   method="backprop", ste_mode="smoothed", sigmas=sigmas, seed=seed,
                                   lr=lr_sste, momentum=mom_sste))
            # Dust K256
            lr_256, mom_256 = _lookup_tuned(tuned, prec, "dust", None, 256, "1M")
            cells.append(RunConfig(task="B", size="L8d512", tokens="1M", precision=prec,
                                   method="dust", population=256, seed=seed, lr=lr_256, momentum=mom_256,
                                   sigmas=sigmas, dust_forward="fake"))

    # K1k at 1 seed (seed 0) [re-cut]
    for prec in precs:
        sigmas = P3_SIGMAS if prec == "w1.58a4" else None
        lr_1k, mom_1k = _lookup_tuned(tuned, prec, "dust", None, 1024, "1M")
        cells.append(RunConfig(task="B", size="L8d512", tokens="1M", precision=prec,
                               method="dust", population=1024, seed=0, lr=lr_1k, momentum=mom_1k,
                               sigmas=sigmas, dust_forward="fake"))
    return cells


def tier_a(seeds: list[int], tuned: dict | None = None) -> list[RunConfig]:
    """Tier A (Plan v3.1 section 6.2): S/10M across P0, P3 x {BPx, K256, K1k}.

    P3 K1k at 1 seed (seed 0) [re-cut].
    Inherits S/1M tuned multipliers scaled to 10M budget defaults.
    """
    cells = []
    for seed in seeds:
        # P0
        lr_fp_bp, mom_fp_bp = _lookup_tuned(tuned, "fp", "backprop", "identity", 0, "10M")
        cells.append(RunConfig(task="A", size="L4d256", tokens="10M", precision="fp",
                               method="backprop", seed=seed, lr=lr_fp_bp, momentum=mom_fp_bp))

        lr_fp_256, mom_fp_256 = _lookup_tuned(tuned, "fp", "dust", None, 256, "10M")
        cells.append(RunConfig(task="A", size="L4d256", tokens="10M", precision="fp",
                               method="dust", population=256, seed=seed, lr=lr_fp_256, momentum=mom_fp_256,
                               dust_forward="fake"))

        lr_fp_1k, mom_fp_1k = _lookup_tuned(tuned, "fp", "dust", None, 1024, "10M")
        cells.append(RunConfig(task="A", size="L4d256", tokens="10M", precision="fp",
                               method="dust", population=1024, seed=seed, lr=lr_fp_1k, momentum=mom_fp_1k,
                               dust_forward="fake"))

        # P3 (w1.58a4)
        lr_p3_ste, mom_p3_ste = _lookup_tuned(tuned, "w1.58a4", "backprop", "identity", 0, "10M")
        cells.append(RunConfig(task="A", size="L4d256", tokens="10M", precision="w1.58a4",
                               method="backprop", ste_mode="identity", seed=seed,
                               lr=lr_p3_ste, momentum=mom_p3_ste))

        lr_p3_sste, mom_p3_sste = _lookup_tuned(tuned, "w1.58a4", "backprop", "smoothed", 0, "10M")
        cells.append(RunConfig(task="A", size="L4d256", tokens="10M", precision="w1.58a4",
                               method="backprop", ste_mode="smoothed", sigmas=P3_SIGMAS, seed=seed,
                               lr=lr_p3_sste, momentum=mom_p3_sste))

        lr_p3_256, mom_p3_256 = _lookup_tuned(tuned, "w1.58a4", "dust", None, 256, "10M")
        cells.append(RunConfig(task="A", size="L4d256", tokens="10M", precision="w1.58a4",
                               method="dust", population=256, seed=seed, lr=lr_p3_256, momentum=mom_p3_256,
                               sigmas=P3_SIGMAS, dust_forward="fake"))

    # P3 K1k at 1 seed (seed 0) [re-cut]
    lr_p3_1k, mom_p3_1k = _lookup_tuned(tuned, "w1.58a4", "dust", None, 1024, "10M")
    cells.append(RunConfig(task="A", size="L4d256", tokens="10M", precision="w1.58a4",
                           method="dust", population=1024, seed=0, lr=lr_p3_1k, momentum=mom_p3_1k,
                           sigmas=P3_SIGMAS, dust_forward="fake"))
    return cells


TIER_GENERATORS = {
    "R": tier_r,
    "tune": tier_tune,
    "B-S": tier_b_s,
    "B-M": tier_b_m,
    "A": tier_a,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="R", choices=list(TIER_GENERATORS.keys()))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default="configs")
    parser.add_argument("--tuned-hparams", default=None,
                        help="path to tune_best_hparams.json (default: results/tune_best_hparams.json if exists)")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    tuned = None
    tuned_path = Path(args.tuned_hparams) if args.tuned_hparams else Path("results/tune_best_hparams.json")
    if tuned_path.is_file():
        print(f"Loading tuned hyperparameters from {tuned_path}")
        tuned = json.loads(tuned_path.read_text())

    generator = TIER_GENERATORS[args.tier]
    cells = generator(seeds, tuned=tuned)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    seen_ids = set()
    for cfg in cells:
        cfg = cfg.resolved()
        if cfg.run_id in seen_ids:
            raise ValueError(f"DUPLICATE run_id generated: {cfg.run_id}")
        seen_ids.add(cfg.run_id)
        path = out / f"{cfg.run_id}.yaml"
        path.write_text(yaml.safe_dump(cfg.to_yaml_dict(), sort_keys=False))
        print(path)
    print(f"\nSUCCESS: {len(cells)} unique configs written to {out}")


if __name__ == "__main__":
    main()
