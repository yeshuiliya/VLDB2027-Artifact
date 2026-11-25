"""Standard external clustering metrics; labels are never used for training."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate_clusters(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("Ground-truth and predicted labels must have matching, nonempty lengths")

    _, tr = np.unique(y_true, return_inverse=True)
    _, pr = np.unique(y_pred, return_inverse=True)
    counts = np.zeros((int(pr.max()) + 1, int(tr.max()) + 1), dtype=np.int64)
    np.add.at(counts, (pr, tr), 1)
    rows, cols = linear_sum_assignment(counts.max() - counts)
    accuracy = float(counts[rows, cols].sum() / len(y_true))
    purity = float(counts.max(axis=1).sum() / len(y_true))
    return {
        "ACC": accuracy,
        "NMI": float(normalized_mutual_info_score(tr, pr)),
        "ARI": float(adjusted_rand_score(tr, pr)),
        "PUR": purity,
    }
