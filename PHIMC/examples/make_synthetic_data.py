#!/usr/bin/env python3
"""Generate a synthetic three-view dataset for multi-view experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-view data")
    parser.add_argument("--output", type=Path, default=Path("data/toy_multiview.npz"))
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    z, labels = make_blobs(n_samples=args.samples, n_features=5,
                            centers=args.classes, cluster_std=0.8, random_state=args.seed)
    rng = np.random.default_rng(args.seed)
    views = {}
    for v, dimension in enumerate((12, 9, 16), start=1):
        transform = rng.normal(size=(5, dimension))
        views[f"X{v}"] = (z @ transform + rng.normal(scale=0.35, size=(args.samples, dimension))).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".mat":
        from scipy.io import savemat
        savemat(args.output, {**views, "Y": labels.reshape(-1,1)})
    elif args.output.suffix.lower() == ".npz":
        np.savez_compressed(args.output, **views, Y=labels)
    else:
        raise ValueError("Please use an .npz or .mat output filename")
    print(f"Created {args.output} | N={args.samples} | views={[v.shape for v in views.values()]}")


if __name__ == "__main__":
    main()
