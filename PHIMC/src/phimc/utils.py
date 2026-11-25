"""Small, conventional utilities for the clustering pipeline."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_minmax(x: np.ndarray) -> np.ndarray:
    """Rescale each sample's feature values independently to [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    low = x.min(axis=1, keepdims=True)
    high = x.max(axis=1, keepdims=True)
    scale = np.where(high > low, high - low, 1.0)
    return (x - low) / scale


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)
