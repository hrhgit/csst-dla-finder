#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from csst_dla.data import make_stratified_split
from csst_dla.fits_utils import read_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a train/validation split.")
    parser.add_argument("--train-fits", default="train.fits")
    parser.add_argument("--out", default="splits/split_seed42.npz")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = read_labels(args.train_fits)
    train_idx, val_idx = make_stratified_split(labels, args.val_fraction, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, train_idx=train_idx, val_idx=val_idx, seed=args.seed)
    print(f"wrote {out}")
    print(f"train: {len(train_idx)}  val: {len(val_idx)}")
    for name, idx in [("train", train_idx), ("val", val_idx)]:
        unique, counts = np.unique(labels["N_DLA"][idx], return_counts=True)
        print(f"{name} N_DLA:", dict(zip(unique.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()

