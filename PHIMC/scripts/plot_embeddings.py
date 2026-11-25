#!/usr/bin/env python3
"""Plot a saved view embedding or fused representation using t-SNE."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize saved clustering embeddings")
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--view", default="fused", help="fused or zero-based index 0,1,...")
    p.add_argument("--perplexity", default=30.0, type=float)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    key = "fused" if args.view == "fused" else f"z_{int(args.view)}"
    with np.load(args.npz, allow_pickle=False) as dataset:
        if key not in dataset:
            raise KeyError(f"Missing {key}; available keys: {sorted(dataset.files)}")
        embedding = np.asarray(dataset[key])
        y = np.asarray(dataset["labels"]).reshape(-1)
    if len(y) < 4:
        raise ValueError("At least four samples are required for t-SNE visualization")
    perplexity = min(args.perplexity, len(y)-1.0)
    coords = TSNE(n_components=2, random_state=args.seed,
                  perplexity=perplexity, init="pca").fit_transform(embedding)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab20", s=12, alpha=0.8)
    ax.set_title(f"t-SNE: {key} (PHIMC)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    dest = args.output or args.npz.with_name(f"tsne_{key}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {dest}")


if __name__ == "__main__":
    main()
