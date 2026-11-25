#!/usr/bin/env python3
"""Run the end-to-end PHIMC clustering pipeline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from phimc.pipeline import cluster, fit_single_view
from phimc.data import load_multiview
from phimc.utils import choose_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PHIMC: view-wise AE + feature fusion + K-means")
    p.add_argument("--dataset", required=True, help="Dataset path, or basename under --data-root")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no-normalize-input", action="store_true")
    p.add_argument("--no-save", action="store_true", help="Do not write results")
    return p.parse_args()


def resolve_dataset(name: str, root: str) -> Path:
    requested = Path(name).expanduser()
    if requested.is_file():
        return requested
    base = Path(root).expanduser()
    candidates = [base / requested, base / (name + ".mat"), base / (name + ".npz")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {name!r}; checked {', '.join(map(str,candidates))}")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.latent_dim <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch-size, latent-dim and hidden-dim must be positive")
    ds_path = resolve_dataset(args.dataset, args.data_root)
    data = load_multiview(ds_path, normalize_input=not args.no_normalize_input)
    clusters = len(np.unique(data.labels))
    device = choose_device(args.device)
    print("\n=== PHIMC | Multi-view Clustering ===")
    print(f"Dataset: {ds_path.name} | N={data.n_samples} | V={data.n_views} | "
          f"dims={data.dimensions} | K={clusters} | device={device}", flush=True)

    rows = []
    output_path = Path(args.output_dir) / ds_path.stem
    if not args.no_save:
        output_path.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"\nSeed {seed}", flush=True)
        set_seed(seed)
        learned = []
        for i, x in enumerate(data.views, start=1):
            print(f"  View {i}/{data.n_views} | shape={x.shape}", flush=True)
            z = fit_single_view(x, device=device, latent_dim=args.latent_dim,
                                hidden_dim=args.hidden_dim, epochs=args.epochs,
                                batch_size=args.batch_size, lr=args.lr)
            learned.append(z)
            fused, pred, score = cluster(learned, data.labels, n_clusters=clusters)
            rows.append({"seed": seed, "view": i, **score})
            print("    " + "  ".join(f"{metric}={value:.4f}" for metric,value in score.items()), flush=True)

        if not args.no_save:
            np.savez_compressed(output_path / f"seed_{seed}.npz",
                                labels=data.labels, predictions=pred, fused=fused,
                                **{f"z_{j}": z for j,z in enumerate(learned)})
    print("\nFinal-view summary (mean ± std across seeds):")
    finals = [r for r in rows if r["view"] == data.n_views]
    for metric in ("ACC", "NMI", "ARI", "PUR"):
        values = np.asarray([r[metric] for r in finals])
        print(f"  {metric}: {values.mean():.4f} ± {values.std():.4f}")
    if not args.no_save:
        with (output_path / "metrics.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["seed", "view", "ACC", "NMI", "ARI", "PUR"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved under {output_path}")


if __name__ == "__main__":
    main()
