"""
svd.py
======
Singular Value Decomposition (SVD) and low-rank matrix approximation.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Dimensionality Reduction > SVD

How it works
------------
Any real matrix A (m x n) can be decomposed as:

    A = U @ diag(S) @ VT

where U (m x k) and VT (k x n) have orthonormal columns/rows and
S (k,) contains the singular values in descending order.

A rank-k approximation is then:

    A_k = U[:, :k] @ diag(S[:k]) @ VT[:k, :]

This is the best rank-k approximation in both the Frobenius and
spectral norms (Eckart-Young theorem).

Usage Example
-------------
>>> from rice_ml.unsupervised.svd import SVD
>>> model = SVD(n_components=20)
>>> model.fit(A)
>>> A_approx = model.reconstruct()
>>> print(model.explained_energy(k=20))
"""

import numpy as np
from typing import Optional


class SVD:
    """
    Truncated SVD via full eigendecomposition of A.T @ A.

    Parameters
    ----------
    n_components : int or None
        Number of singular values/vectors to retain. If None, keeps all.

    Attributes
    ----------
    U_  : ndarray of shape (m, k)
    S_  : ndarray of shape (k,)   — singular values, descending
    VT_ : ndarray of shape (k, n)
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.U_:  Optional[np.ndarray] = None
        self.S_:  Optional[np.ndarray] = None
        self.VT_: Optional[np.ndarray] = None

    def fit(self, A: np.ndarray) -> "SVD":
        """
        Compute the (truncated) SVD of matrix A.

        Parameters
        ----------
        A : array-like of shape (m, n)
        """
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D, got shape {A.shape}")

        # numpy's SVD — full_matrices=False gives the economy/thin form
        U, S, VT = np.linalg.svd(A, full_matrices=False)

        k = self.n_components if self.n_components is not None else len(S)
        k = min(k, len(S))

        self.U_  = U[:, :k]
        self.S_  = S[:k]
        self.VT_ = VT[:k, :]
        return self

    def reconstruct(self, k: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct (approximate) the original matrix using the top-k components.

        Parameters
        ----------
        k : int or None — defaults to n_components used at fit time.
        """
        self._check_fitted()
        k = k if k is not None else len(self.S_)
        k = min(k, len(self.S_))
        return self.U_[:, :k] @ np.diag(self.S_[:k]) @ self.VT_[:k, :]

    def frobenius_error(self, A: np.ndarray, k: Optional[int] = None) -> float:
        """||A - A_k||_F — reconstruction error at rank k."""
        return float(np.linalg.norm(np.asarray(A, float) - self.reconstruct(k), "fro"))

    def explained_energy(self, k: Optional[int] = None) -> float:
        """
        Fraction of total energy (sum of squared singular values) captured
        by the top-k components.
        """
        self._check_fitted()
        k = k if k is not None else len(self.S_)
        total = float(np.sum(self.S_ ** 2))
        return float(np.sum(self.S_[:k] ** 2)) / total if total > 0 else 0.0

    def cumulative_energy(self) -> np.ndarray:
        """
        Cumulative energy fraction for each singular value — useful for
        choosing k (e.g., find where this crosses 0.95).
        """
        self._check_fitted()
        sq = self.S_ ** 2
        return np.cumsum(sq) / sq.sum()

    def _check_fitted(self):
        if self.S_ is None:
            raise RuntimeError("Call fit(A) first.")
