# PHIMC

**Sequential Multi-View Representation Learning and Clustering**

PHIMC provides a runnable Python workflow for clustering aligned multi-view data as views are processed sequentially. The code includes view-specific autoencoder training with reconstruction loss, feature concatenation, K-means clustering, evaluation metrics, and embedding visualization.

## Features

- Load multi-view `.npz` and MATLAB `.mat` data, including MATLAB v7.3/HDF5 cell arrays.
- Train a representation for each view with a reconstruction-based autoencoder.
- Concatenate equal-weight representations from the processed views and run K-means.
- Report per-stage ACC, NMI, ARI, and Purity across user-specified random seeds.
- Export clustering results and embeddings for subsequent analysis and t-SNE visualization.
- Generate synthetic multi-view data for reproducible installation checks.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run

Create synthetic multi-view data and run clustering:

```bash
python examples/make_synthetic_data.py --output data/synthetic_multiview.npz
python scripts/run_phimc.py --dataset data/synthetic_multiview.npz --epochs 5 --seeds 0
python scripts/plot_embeddings.py --npz outputs/synthetic_multiview/seed_0.npz --view fused
```

CPU and CUDA are supported (`--device auto` chooses automatically).

## Your own datasets

Each view is an `N x D_v` matrix representing the same `N` samples in the same row order. Supported layouts:

```text
X1: (N, D1)
X2: (N, D2)
X3: (N, D3)   # optional
Y:  (N,)      # ground-truth labels for evaluation
```

MATLAB cell-array layout `X` together with `Y` is supported, including HDF5-based MATLAB v7.3 files. Transposed matrices `(D_v, N)` are detected using label length. See [data_format.md](docs/data_format.md).

```bash
python scripts/run_phimc.py --dataset /path/to/dataset.mat --epochs 100 --seeds 0 1 2 3 4
# or look up a dataset basename under data/
python scripts/run_phimc.py --data-root data --dataset Handwritten
```

### Main options

| Option | Meaning | Default |
| --- | --- | --- |
| `--dataset` | Input `.mat`/`.npz` path or dataset basename | required |
| `--data-root` | Dataset lookup directory | `data` |
| `--latent-dim` | Latent representation dimension | `128` |
| `--hidden-dim` | Autoencoder hidden dimension | `256` |
| `--epochs` | Training epochs per view | `100` |
| `--batch-size` | Training batch size | `256` |
| `--lr` | Learning rate | `1e-4` |
| `--seeds` | Repeated experiment seeds | `0` |
| `--device` | `auto`, `cpu`, `cuda` | `auto` |
| `--output-dir` | Results directory | `outputs` |

## Repository structure

```text
PHIMC/
├── README.md
├── README_CN.md
├── pyproject.toml
├── requirements.txt
├── src/phimc/
│   ├── data.py         # multi-view data loading
│   ├── model.py        # view-specific autoencoder
│   ├── pipeline.py     # training, feature fusion, clustering
│   ├── metrics.py      # clustering evaluation
│   └── utils.py        # preprocessing and reproducibility
├── scripts/
│   ├── run_phimc.py
│   └── plot_embeddings.py
├── examples/
│   └── make_synthetic_data.py
├── docs/
│   └── data_format.md
└── tests/
    └── test_pipeline.py
```

## Results

The main script prints each view's metrics and the mean/std across seeds. By default, results are written to `outputs/<dataset>/`:

- `metrics.csv` — per-view clustering metrics.
- `seed_<seed>.npz` — labels, predictions, fused representation, and individual view representations (`z_0`, `z_1`, ...).

To visualize a particular view:

```bash
python scripts/plot_embeddings.py --npz outputs/synthetic_multiview/seed_0.npz --view 0
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

Ground-truth labels are used only to determine the evaluation cluster count and calculate metrics; autoencoder reconstruction is unsupervised. The fusion stage gives equal contributions to the processed views. Input features can be rescaled by column to `[0, 1]`. Generated data, model checkpoints, and experimental outputs are excluded from version control by `.gitignore`. Publish only datasets that you have permission to redistribute.
