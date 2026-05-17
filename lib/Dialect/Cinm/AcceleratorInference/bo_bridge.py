"""
Python bridge module called from C++ via pybind11.

Public API
----------
lhs_indices(encoded, n, seed) -> list[int]
    Select n indices from the rows of encoded (N×D float32 ndarray)
    using Latin Hypercube Sampling.  Returns a list of integer indices.

next_candidate_indices(X_obs, y_obs, X_pool, k, kappa, epochs,
                       n_ensemble, hidden, depth) -> list[int]
    Fit a BANANAS MLP ensemble on (X_obs, y_obs), score the rows of X_pool
    by UCB acquisition (minimisation), and return the k indices with the
    lowest UCB scores.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ── Surrogate ─────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers: list = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _BananasEnsemble:
    """Ensemble of MLPs trained with bootstrap resampling (BANANAS-style)."""

    def __init__(self, in_dim: int, n: int = 5, hidden: int = 64, depth: int = 2):
        self.models = [_MLP(in_dim, hidden, depth) for _ in range(n)]
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> None:
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y_norm, dtype=torch.float32)
        n = len(X)
        for model in self.models:
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
            idx = np.random.choice(n, n, replace=True)
            Xb, yb = Xt[idx], yt[idx]
            for _ in range(epochs):
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(Xb), yb)
                loss.backward()
                opt.step()
                sched.step()

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xt = torch.tensor(X, dtype=torch.float32)
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(Xt).numpy())
        stacked = np.stack(preds)  # (n_models, N)
        mu = stacked.mean(axis=0) * self._y_std + self._y_mean
        sigma = stacked.std(axis=0) * self._y_std
        return mu, sigma


# ── LHS ───────────────────────────────────────────────────────────────────────

def lhs_indices(encoded: np.ndarray, n: int, seed: int = 42) -> list[int]:
    """
    Select *n* rows from *encoded* (shape N×D) via Latin Hypercube Sampling.
    Returns a list of integer row-indices (length ≤ n).
    """
    rng = np.random.default_rng(seed)
    N, D = encoded.shape
    n = min(n, N)
    if n == 0:
        return []

    mins = encoded.min(axis=0)
    maxs = encoded.max(axis=0)
    ranges = np.where(maxs > mins, maxs - mins, 1.0)
    normed = (encoded - mins) / ranges  # (N, D) in [0, 1]

    # LHS target points — one per stratum per dimension
    lhs = np.zeros((n, D))
    for d in range(D):
        lhs[:, d] = (rng.permutation(n) + rng.uniform(size=n)) / n

    # Greedy nearest-neighbour assignment
    remaining = list(range(N))
    selected: list[int] = []
    for target in lhs:
        if not remaining:
            break
        dists = np.linalg.norm(normed[remaining] - target, axis=1)
        best_pos = int(np.argmin(dists))
        selected.append(remaining[best_pos])
        remaining.pop(best_pos)
    return selected


# ── Surrogate-guided next-batch ────────────────────────────────────────────────

def next_candidate_indices(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_pool: np.ndarray,
    k: int = 1,
    kappa: float = 2.0,
    epochs: int = 200,
    n_ensemble: int = 5,
    hidden: int = 64,
    depth: int = 2,
) -> list[int]:
    """
    Fit a BANANAS ensemble on (X_obs, y_obs), score X_pool by UCB (for
    minimisation: score = mu − κ·σ, lower is better), and return the k
    indices with the lowest scores.

    Parameters
    ----------
    X_obs   : (n_obs, D) float32 — encoded observed configurations
    y_obs   : (n_obs,)   float32 — observed objective values
    X_pool  : (n_pool, D) float32 — encoded unvisited candidates
    k       : number of candidates to return
    kappa   : exploration weight (higher → more exploration)
    epochs  : training epochs per MLP
    n_ensemble, hidden, depth : ensemble architecture
    """
    in_dim = X_obs.shape[1]
    ensemble = _BananasEnsemble(in_dim, n=n_ensemble, hidden=hidden, depth=depth)
    ensemble.fit(X_obs, y_obs, epochs=epochs)
    mu, sigma = ensemble.predict(X_pool)
    # UCB for minimisation: prefer low mean AND high uncertainty
    scores = mu - kappa * sigma
    top_k = np.argsort(scores)[:k]
    return top_k.tolist()
