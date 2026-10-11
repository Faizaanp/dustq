"""CLI for the idempotent run queue: python scripts/queue.py [--configs-dir configs] [--only substr]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dustq.queue import run_queue  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--only", help="run only configs whose run_id contains this substring")
    parser.add_argument("--cu-burn-rate", default="", help="CU/h observed in the Colab Resources tab")
    args = parser.parse_args()
    for line in run_queue(args.configs_dir, args.results_dir, only=args.only,
                          cu_burn_rate=args.cu_burn_rate):
        print(line, flush=True)


if __name__ == "__main__":
    main()
