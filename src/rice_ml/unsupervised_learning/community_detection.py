"""
community_detection.py
======================
Graph community detection via Label Propagation.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Unsupervised Learning > Community Detection

How it works
------------
Each node starts with a unique label. At every iteration each node adopts
the label that is most *weighted* among its neighbours (sum of edge weights).
The process repeats until labels stabilise or max_iter is reached.

Graphs are passed as NumPy adjacency matrices. Edge weights are used
automatically — unweighted graphs can use a binary (0/1) matrix.

Usage Example
-------------
>>> from rice_ml.unsupervised.community_detection import LabelPropagation
>>> lpa = LabelPropagation(max_iter=50, random_state=42)
>>> lpa.fit(A)          # A is an (n, n) adjacency matrix
>>> print(lpa.labels_)
>>> print(lpa.n_communities_)
"""

import numpy as np
from typing import Optional


class LabelPropagation:
    """
    Weighted Label Propagation for community detection.

    Parameters
    ----------
    max_iter     : int — maximum propagation rounds, default=100
    random_state : int or None — seed for node-order shuffling

    Attributes
    ----------
    labels_        : ndarray (n_nodes,) — community label per node
    n_communities_ : int — number of distinct communities found
    n_iter_        : int — iterations actually run
    """

    def __init__(self, max_iter: int = 100, random_state: Optional[int] = None):
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_:        Optional[np.ndarray] = None
        self.n_communities_: int = 0
        self.n_iter_:        int = 0

    def fit(self, A: np.ndarray) -> "LabelPropagation":
        """
        Run label propagation on an adjacency matrix.

        Parameters
        ----------
        A : ndarray (n, n) — weighted or binary adjacency matrix (undirected)
        """
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square 2-D adjacency matrix.")

        n = A.shape[0]
        labels = np.arange(n)
        rng = np.random.default_rng(self.random_state)

        for iteration in range(self.max_iter):
            old_labels = labels.copy()
            for node in rng.permutation(n):       # random order prevents bias
                nbrs = np.where(A[node] > 0)[0]
                if len(nbrs) == 0:
                    continue
                # Weighted vote: sum edge weights per neighbour label
                scores: dict = {}
                for nb, w in zip(nbrs, A[node, nbrs]):
                    lbl = labels[nb]
                    scores[lbl] = scores.get(lbl, 0.0) + w
                labels[node] = max(scores, key=scores.get)

            self.n_iter_ = iteration + 1
            if np.array_equal(labels, old_labels):
                break

        self.labels_        = labels
        self.n_communities_ = int(len(np.unique(labels)))
        return self

    def fit_predict(self, A: np.ndarray) -> np.ndarray:
        """Fit and return community labels."""
        return self.fit(A).labels_.copy()
