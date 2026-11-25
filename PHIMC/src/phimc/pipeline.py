"""Sequential view-specific autoencoder training and equal-weight clustering."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F

from .metrics import evaluate_clusters
from .model import Autoencoder
from .utils import row_minmax


def fit_single_view(x: np.ndarray, *, device: torch.device, latent_dim: int,
                    hidden_dim: int, epochs: int, batch_size: int,
                    lr: float, verbose: bool = True) -> np.ndarray:
    """Learn an embedding for one view using reconstruction error."""
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    model = Autoencoder(x.shape[1], latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n, device=device)
        for offset in range(0, n, batch_size):
            batch = x_tensor[order[offset:offset + batch_size]]
            _, reconstruction = model(batch)
            loss = F.mse_loss(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(f"    epoch {epoch:>3}/{epochs} | reconstruction={loss.item():.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        return model.encoder(x_tensor).cpu().numpy().astype(np.float32)


def equal_fusion(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one view is required")
    if len({x.shape[0] for x in embeddings}) != 1:
        raise ValueError("All views must share the same sample count")
    combined = np.concatenate([row_minmax(z) for z in embeddings], axis=1)
    return row_minmax(combined)


def cluster(embeddings: list[np.ndarray], y_true: np.ndarray,
            *, n_clusters: int, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, dict]:
    fused = equal_fusion(embeddings)
    predictions = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300,
                         random_state=random_state).fit_predict(fused)
    metrics = evaluate_clusters(y_true, predictions)
    return fused, predictions, metrics
