"""
kmeans.py
=========
K-Means clustering from scratch.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Unsupervised Learning > Clustering

Usage Example
-------------
>>> from rice_ml.unsupervised.kmeans import KMeans
>>> model = KMeans(k=5, random_state=42)
>>> model.fit(X_scaled)
>>> print(model.inertia_)
"""

import numpy as np
from typing import Optional


class KMeans:
    """
    K-Means clustering with k-means++ initialisation.

    Parameters
    ----------
    k            : int   — number of clusters, default=3
    max_iters    : int   — iteration limit, default=100
    tol          : float — centroid shift tolerance for early stop, default=1e-4
    random_state : int or None

    Attributes
    ----------
    centroids_ : ndarray (k, n_features)
    labels_    : ndarray (n_samples,)
    inertia_   : float — sum of squared distances to nearest centroid
    """

    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None
        self.labels_:    Optional[np.ndarray] = None
        self.inertia_:   Optional[float] = None

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-Means++ initialisation for better convergence."""
        idx = rng.integers(0, len(X))
        centroids = [X[idx]]
        for _ in range(self.k - 1):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            centroids.append(X[rng.choice(len(X), p=probs)])
        return np.array(centroids)

    def _assign(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(dists, axis=1)

    def _inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        return float(sum(
            np.sum((X[labels == j] - self.centroids_[j]) ** 2)
            for j in range(self.k)
        ))

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}")

        rng = np.random.default_rng(self.random_state)
        self.centroids_ = self._init_centroids(X, rng)

        for _ in range(self.max_iters):
            labels = self._assign(X)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else self.centroids_[j]
                for j in range(self.k)
            ])
            if np.linalg.norm(new_centroids - self.centroids_) < self.tol:
                break
            self.centroids_ = new_centroids

        self.labels_  = self._assign(X)
        self.inertia_ = self._inertia(X, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Call fit() first.")
        return self._assign(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray) -> float:
        """Returns negative inertia (higher is better, sklearn convention)."""
        X = np.asarray(X, dtype=float)
        labels = self.predict(X)
        return -self._inertia(X, labels)
