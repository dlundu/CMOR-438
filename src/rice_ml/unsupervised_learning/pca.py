"""
pca.py
======
Principal Component Analysis (PCA) via eigendecomposition of the
covariance matrix.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Dimensionality Reduction > PCA

How it works
------------
1. Center the data: X_c = X - mean(X)
2. Compute covariance matrix: C = X_c.T @ X_c / (n - 1)
3. Eigendecompose C → eigenvalues λ, eigenvectors W
4. Sort by λ descending — eigenvectors are the principal components.
5. Project: Z = X_c @ W[:, :k]

Usage Example
-------------
>>> from rice_ml.unsupervised.pca import PCA
>>> pca = PCA(n_components=2)
>>> Z = pca.fit_transform(X_scaled)
>>> print(pca.explained_variance_ratio_)
"""

import numpy as np
from typing import Optional, Union


class PCA:
    """
    Principal Component Analysis via covariance matrix eigendecomposition.

    Parameters
    ----------
    n_components : int, float, or None
        - int   : exact number of components to keep.
        - float : retain enough components to explain this fraction of variance
                  (e.g. 0.95 keeps components until ≥95% variance explained).
        - None  : keep all components.

    Attributes
    ----------
    components_              : ndarray (n_components, n_features)
    explained_variance_ratio_: ndarray (n_components,)
    mean_                    : ndarray (n_features,)
    n_components_            : int
    """

    def __init__(self, n_components: Optional[Union[int, float]] = None):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        cov = np.cov(X_c, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        ratio = eigenvalues / eigenvalues.sum()

        # Resolve n_components
        if self.n_components is None:
            k = n_features
        elif isinstance(self.n_components, float):
            if not 0.0 < self.n_components < 1.0:
                raise ValueError("Float n_components must be in (0, 1).")
            k = int(np.argmax(np.cumsum(ratio) >= self.n_components) + 1)
        elif isinstance(self.n_components, int):
            if not 1 <= self.n_components <= n_features:
                raise ValueError(f"n_components must be in [1, {n_features}].")
            k = self.n_components
        else:
            raise TypeError(f"Invalid n_components type: {type(self.n_components)}")

        self.components_               = eigenvectors[:, :k].T   # (k, n_features)
        self.explained_variance_ratio_ = ratio[:k]
        self.n_components_             = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call fit() first.")
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call fit() first.")
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_
