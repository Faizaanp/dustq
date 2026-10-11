"""T2 data verification (Plan v3.1 section 7, T2): shapes, ranges, disjointness, manifest.

Run after prepare_data.py: python scripts/check_data.py --data-dir data
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def row_hashes(tensor):
    return {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in tensor}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    d = Path(args.data_dir)
    expected = {"train_1M.pt": 488, "train_10M.pt": 4880, "val.pt": 544, "test.pt": 544}
    splits, problems = {}, []
    for name, rows in expected.items():
        path = d / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=True)
        assert data.ndim == 2 and data.shape[1] == 2049, f"{name}: bad shape {tuple(data.shape)}"
        if len(data) != rows:
            problems.append(f"{name}: expected {rows} rows, found {len(data)}")
        if data.dtype not in (torch.int32, torch.int64):
            problems.append(f"{name}: non-integer dtype {data.dtype}")
        if int(data.min()) < 0 or int(data.max()) >= 4096:
            problems.append(f"{name}: token ids out of [0, 4096)")
        print(f"{name}: {len(data)} rows x {data.shape[1]} (prediction tokens: {len(data) * 2048:,})")
        splits[name] = data

    if not problems:
        # Disjointness where it is REQUIRED: the training pool vs held-out splits, and val vs test.
        # (train_1M is BY DESIGN a strict prefix of train_10M — same pool, different budget.)
        pool = row_hashes(splits["train_10M.pt"])
        for other in ("val.pt", "test.pt"):
            overlap = pool & row_hashes(splits[other])
            if overlap:
                problems.append(f"{len(overlap)} row(s) shared between train_10M and {other}")
        vt = row_hashes(splits["val.pt"]) & row_hashes(splits["test.pt"])
        if vt:
            problems.append(f"{len(vt)} row(s) shared between val and test")
        # Prefix property: train_1M must be the first 488 rows of train_10M.
        if "train_1M.pt" in splits and "train_10M.pt" in splits:
            same = torch.equal(splits["train_10M.pt"][:488], splits["train_1M.pt"])
            print(f"train_1M is a strict prefix of train_10M: {same}")
            if not same:
                problems.append("train_1M is not a prefix of train_10M")

    manifest = d / "manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        for name, info in m.get("files", {}).items():
            actual = hashlib.sha256((d / name).read_bytes()).hexdigest()
            ok = actual == info["sha256"]
            print(f"manifest {name}: sha256 {'matches' if ok else 'MISMATCH'}")
            if not ok:
                problems.append(f"manifest hash mismatch for {name}")
        print(f"corpus: {m.get('corpus')} / {m.get('subset')} / skipped {m.get('skipped_documents')} docs")
    else:
        problems.append("manifest.json missing")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nAll data checks passed.")


if __name__ == "__main__":
    main()
