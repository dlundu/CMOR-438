"""
dbscan.py
=========
Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Unsupervised Learning > Clustering

How it works
------------
A point is a *core point* if it has at least min_samples neighbours within
radius eps. Core points seed clusters; border points are reachable from a
core point but aren't core themselves. Everything else is noise (label -1).

Usage Example
-------------
>>> from rice_ml.unsupervised.dbscan import DBSCAN
>>> model = DBSCAN(eps=1.0, min_samples=5)
>>> model.fit(X_scaled)
>>> print(model.labels_)       # -1 = noise
>>> print(model.n_clusters_)
"""

import numpy as np
from typing import Optional


class DBSCAN:
    """
    DBSCAN clustering.

    Parameters
    ----------
    eps         : float — neighbourhood radius, default=0.5
    min_samples : int   — minimum neighbours to be a core point, default=5

    Attributes
    ----------
    labels_    : ndarray (n_samples,) — cluster ids; -1 = noise
    n_clusters_: int — number of clusters found (excludes noise)
    """

    _UNVISITED = 0
    _NOISE     = -1

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        if eps <= 0:
            raise ValueError("eps must be > 0.")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_:     Optional[np.ndarray] = None
        self.n_clusters_: int = 0

    def _neighbors(self, X: np.ndarray, idx: int) -> np.ndarray:
        """Indices of all points within eps of X[idx]."""
        diff = X - X[idx]
        return np.flatnonzero(np.sum(diff * diff, axis=1) <= self.eps ** 2)

    def _expand(self, X: np.ndarray, labels: np.ndarray, idx: int, cluster_id: int):
        """BFS expansion from a core point to build a cluster."""
        labels[idx] = cluster_id
        queue = list(self._neighbors(X, idx))
        head = 0
        while head < len(queue):
            pt = queue[head]; head += 1
            if labels[pt] in (self._UNVISITED, self._NOISE):
                labels[pt] = cluster_id
                nbrs = self._neighbors(X, pt)
                if len(nbrs) >= self.min_samples:
                    for nb in nbrs:
                        if labels[nb] == self._UNVISITED:
                            queue.append(nb)

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}")

        n = len(X)
        labels = np.full(n, self._UNVISITED, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != self._UNVISITED:
                continue
            nbrs = self._neighbors(X, i)
            if len(nbrs) >= self.min_samples:
                cluster_id += 1
                self._expand(X, labels, i, cluster_id)
            else:
                labels[i] = self._NOISE

        # Re-label to sklearn convention: 0-based cluster ids, -1 for noise
        self.labels_ = np.where(labels == self._NOISE, -1, labels - 1)
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
