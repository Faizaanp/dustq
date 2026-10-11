"""Emit the run configs for a tier into configs/ (one YAML per run, named by run_id).

Tier R (Plan v3.1 section 6.2, gate G0):
  M/1M/P0: backprop + Dust K in {256, 1k, 4k}, 3 seeds each;
  S/10M/P0: backprop + Dust K=256, 3 seeds each.

Usage: python scripts/make_configs.py --tier R [--seeds 0,1,2] [--out configs]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dustq.config import RunConfig  # noqa: E402


def tier_r(seeds):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="R", choices=["R"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default="configs")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = tier_r(seeds) if args.tier == "R" else []
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for cfg in cells:
        cfg = cfg.resolved()
        path = out / f"{cfg.run_id}.yaml"
        path.write_text(yaml.safe_dump(cfg.to_yaml_dict(), sort_keys=False))
        print(path)
    print(f"{len(cells)} configs written to {out}")


if __name__ == "__main__":
    main()
