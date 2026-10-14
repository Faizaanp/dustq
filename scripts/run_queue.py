"""CLI for the idempotent run queue: python scripts/run_queue.py [--configs-dir configs] [--only substr]

Named run_queue.py (not queue.py) to avoid shadowing the stdlib `queue` module, which
torch imports — recorded as a deviation from the Plan v3.1 file map.
"""

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
    parser.add_argument("--retry-failed", action="store_true",
                        help="reset FAILED_ATTEMPTS markers for matching runs and retry them")
    args = parser.parse_args()
    for line in run_queue(args.configs_dir, args.results_dir, only=args.only,
                          cu_burn_rate=args.cu_burn_rate, retry_failed=args.retry_failed):
        print(line, flush=True)


if __name__ == "__main__":
    main()
