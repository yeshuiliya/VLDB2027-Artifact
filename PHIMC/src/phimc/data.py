"""Generic loaders for multi-view arrays. Aligned views and labels support .mat and .npz input."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class MultiViewData:
    views: list[np.ndarray]
    labels: np.ndarray

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def dimensions(self) -> list[int]:
        return [x.shape[1] for x in self.views]


def _numbered_view_keys(keys) -> list[str]:
    result = []
    for key in keys:
        found = re.fullmatch(r"X([1-9][0-9]*)", str(key), re.IGNORECASE)
        if found:
            result.append((int(found.group(1)), key))
    return [key for _, key in sorted(result)]


def _label_key(keys) -> str:
    lookup = {str(k).lower(): k for k in keys}
    for name in ("y", "labels", "label", "gnd", "gt"):
        if name in lookup:
            return lookup[name]
    raise ValueError("No labels found; expected Y, labels, gnd, or gt")


def _orient(x: np.ndarray, n_samples: int, description: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.dtype.kind not in "fiub":
        raise ValueError(f"{description} must be a two-dimensional numeric matrix")
    if x.shape[0] == n_samples:
        return x.astype(np.float32)
    if x.shape[1] == n_samples:
        return x.T.astype(np.float32)
    raise ValueError(f"{description} shape {x.shape} incompatible with N={n_samples}")


def _extract_standard(mapping) -> tuple[list, np.ndarray]:
    keys = list(mapping.keys())
    labels = np.asarray(mapping[_label_key(keys)]).reshape(-1)
    x_keys = _numbered_view_keys(keys)
    if x_keys:
        return [np.asarray(mapping[k]) for k in x_keys], labels
    if "X" in mapping:
        cells = np.asarray(mapping["X"])
        if cells.dtype == object:
            return [np.asarray(c) for c in cells.reshape(-1)], labels
    raise ValueError("No multi-view features found; expected X1, X2, ... or MATLAB cell X")


def _extract_hdf5(path: Path) -> tuple[list, np.ndarray]:
    with h5py.File(path, "r") as handle:
        keys = [k for k in handle.keys() if not k.startswith("#")]
        labels = np.asarray(handle[_label_key(keys)]).reshape(-1)
        view_keys = _numbered_view_keys(keys)
        if view_keys:
            return [np.asarray(handle[k]) for k in view_keys], labels
        if "X" not in handle:
            raise ValueError("HDF5 .mat has no X or X1..Xn")
        cells = np.asarray(handle["X"])
        if h5py.check_dtype(ref=cells.dtype) is None and cells.dtype != object:
            raise ValueError("HDF5 X is not a MATLAB cell array of references")
        views = []
        for ref in cells.reshape(-1):
            if not ref:
                raise ValueError("Empty reference in HDF5 X cell array")
            views.append(np.asarray(handle[ref]))
        return views, labels


def load_multiview(path: str | Path, normalize_input: bool = True) -> MultiViewData:
    """Load aligned feature views from `.mat` (classic/v7.3) or `.npz`.

    Labels are used only to infer evaluation cluster count and report metrics.
    The column-wise min-max input preprocessing is an optional
    convention; change it if your dataset already has standardized features.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            raw_views, labels = _extract_standard(archive)
    elif suffix == ".mat":
        try:
            mat = scipy.io.loadmat(path)
            raw_views, labels = _extract_standard(mat)
        except (NotImplementedError, ValueError) as exc:
            if not h5py.is_hdf5(path):
                raise
            raw_views, labels = _extract_hdf5(path)
    else:
        raise ValueError("Supported formats: .mat and .npz")

    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("Empty label vector")
    if not raw_views:
        raise ValueError("No views found")
    views = []
    for i, view in enumerate(raw_views, start=1):
        x = _orient(view, len(labels), f"View {i}")
        if not np.isfinite(x).all():
            raise ValueError(f"View {i} contains NaN or Inf")
        if normalize_input:
            x = MinMaxScaler().fit_transform(x).astype(np.float32)
        views.append(x)

    # Use dense integer labels so metrics are insensitive to MATLAB 1-based IDs.
    _, labels_int = np.unique(labels, return_inverse=True)
    return MultiViewData(views=views, labels=labels_int.astype(np.int64))
