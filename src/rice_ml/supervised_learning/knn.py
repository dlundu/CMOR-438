"""
From-scratch implementation of K-Nearest Neighbors for both classification
and regression using NumPy.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Supervised Learning > Nearest Neighbors

How it works
------------
KNN is a non-parametric, lazy learner — it stores the entire training set
and defers all computation to prediction time. To classify (or regress) a
new point q:

    1. Compute the distance from q to every training point.
    2. Identify the k closest neighbors.
    3. Classification : take the majority class vote (or distance-weighted).
       Regression     : take the mean target value (or distance-weighted mean).

Distance metrics supported
--------------------------
- 'euclidean' : sqrt(sum((a - b)^2))   — default, sensitive to scale
- 'manhattan' : sum(|a - b|)           — less sensitive to outliers

Usage Example
-------------
>>> from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor
>>> clf = KNNClassifier(n_neighbors=5)
>>> clf.fit(X_train, y_train)
>>> print(clf.score(X_test, y_test))

>>> reg = KNNRegressor(n_neighbors=7, weights='distance')
>>> reg.fit(X_train, y_train)
>>> print(reg.score(X_test, y_test))
"""

import numpy as np
from typing import Literal, Optional, Tuple


def _pairwise_distances(
    XA: np.ndarray,
    XB: np.ndarray,
    metric: str,
) -> np.ndarray:
    """
    Compute an (n_query, n_train) distance matrix between two sets of points.

    Uses vectorised operations throughout — no Python loops over samples.

    Parameters
    ----------
    XA : ndarray of shape (n_query, n_features)
    XB : ndarray of shape (n_train, n_features)
    metric : {'euclidean', 'manhattan'}

    Returns
    -------
    D : ndarray of shape (n_query, n_train)
    """
    if metric == "euclidean":
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b  (avoids an explicit broadcast loop)
        sq_a = np.sum(XA ** 2, axis=1, keepdims=True)   # (n_query, 1)
        sq_b = np.sum(XB ** 2, axis=1, keepdims=True).T  # (1, n_train)
        D_sq = np.maximum(sq_a + sq_b - 2.0 * XA @ XB.T, 0.0)
        return np.sqrt(D_sq)
    elif metric == "manhattan":
        # Explicit broadcast: (n_query, 1, d) - (1, n_train, d)
        return np.sum(np.abs(XA[:, None, :] - XB[None, :, :]), axis=2)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose 'euclidean' or 'manhattan'.")


def _neighbor_weights(distances: np.ndarray, scheme: str, eps: float = 1e-12) -> np.ndarray:
    """
    Convert a distance matrix to neighbor weights.

    Parameters
    ----------
    distances : ndarray of shape (n_query, k)
    scheme    : 'uniform' — equal weight for every neighbor
                'distance' — inverse-distance weighting; exact ties get weight 1
    eps       : small floor to avoid division by zero

    Returns
    -------
    weights : ndarray of shape (n_query, k), rows sum to 1 after normalisation
    """
    if scheme == "uniform":
        return np.ones_like(distances)

    # If any neighbor is an exact hit (dist ≈ 0), give it all the weight
    exact_hit = distances <= eps                        # (n_query, k) bool
    has_hit   = exact_hit.any(axis=1, keepdims=True)   # (n_query, 1) bool

    inv = 1.0 / np.maximum(distances, eps)
    w = np.where(has_hit, exact_hit.astype(float), inv)
    return w

class _KNNBase:
    """
    Stores training data and exposes kneighbors() used by both subclasses.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"metric must be 'euclidean' or 'manhattan', got '{metric}'.")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got '{weights}'.")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_KNNBase":
        """Store the training data (KNN has no explicit training step)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")
        if self.n_neighbors > len(X):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) cannot exceed the number of "
                f"training samples ({len(X)})."
            )

        self._X_train = X
        self._y_train = y
        return self

    def kneighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k nearest training neighbors for each query point.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)

        Returns
        -------
        distances : ndarray of shape (n_query, k), sorted ascending
        indices   : ndarray of shape (n_query, k), into the training set
        """
        if self._X_train is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"Feature count mismatch: training had {self._X_train.shape[1]} "
                f"features, query has {X.shape[1]}."
            )

        D = _pairwise_distances(X, self._X_train, self.metric)  # (n_query, n_train)

        # argpartition is O(n) — cheaper than full argsort when k << n_train
        k = self.n_neighbors
        part_idx = np.argpartition(D, k - 1, axis=1)[:, :k]    # (n_query, k) unordered

        # Sort only the k selected columns
        part_dist = np.take_along_axis(D, part_idx, axis=1)
        order = part_dist.argsort(axis=1)
        indices   = np.take_along_axis(part_idx,  order, axis=1)
        distances = np.take_along_axis(part_dist, order, axis=1)

        return distances, indices

class KNNClassifier(_KNNBase):
    """
    K-Nearest Neighbors Classifier.

    Predicts the class of each query point by majority vote (uniform) or
    distance-weighted vote among its k nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric.
    weights : {'uniform', 'distance'}, default='uniform'
        'uniform'  — each neighbor gets one vote.
        'distance' — closer neighbors have proportionally more influence.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels seen during fit, in sorted order.

    Examples
    --------
    >>> from rice_ml.supervised.knn import KNNClassifier
    >>> clf = KNNClassifier(n_neighbors=5)
    >>> clf.fit(X_train, y_train)
    >>> clf.score(X_test, y_test)
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        super().fit(X, y)
        self.classes_ = np.unique(self._y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probability estimates via weighted neighbor voting.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)

        Returns
        -------
        proba : ndarray of shape (n_query, n_classes)
            Rows sum to 1.
        """
        distances, indices = self.kneighbors(X)
        w = _neighbor_weights(distances, self.weights)      # (n_query, k)
        neighbor_labels = self._y_train[indices]            # (n_query, k)

        # Map each neighbor label to a class index, then accumulate weighted votes
        n_classes = len(self.classes_)
        class_idx = np.searchsorted(self.classes_, neighbor_labels)  # (n_query, k)
        one_hot   = np.eye(n_classes)[class_idx]                     # (n_query, k, n_classes)
        proba     = (one_hot * w[..., None]).sum(axis=1)             # (n_query, n_classes)
        proba    /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Returns
        -------
        y_pred : ndarray of shape (n_query,)
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

class KNNRegressor(_KNNBase):
    """
    K-Nearest Neighbors Regressor.

    Predicts continuous target values as the (weighted) mean of the k
    nearest neighbors' target values.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric.
    weights : {'uniform', 'distance'}, default='uniform'
        'uniform'  — simple mean of neighbor targets.
        'distance' — inverse-distance weighted mean.

    Examples
    --------
    >>> from rice_ml.supervised_learning.knn import KNNRegressor
    >>> reg = KNNRegressor(n_neighbors=7, weights='distance')
    >>> reg.fit(X_train, y_train)
    >>> reg.score(X_test, y_test)
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        super().fit(X, y)
        self._y_train = self._y_train.astype(float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.

        Returns
        -------
        y_pred : ndarray of shape (n_query,)
        """
        distances, indices = self.kneighbors(X)
        w = _neighbor_weights(distances, self.weights)   # (n_query, k)
        neighbor_targets = self._y_train[indices]        # (n_query, k)

        w_sum = w.sum(axis=1)
        return (w * neighbor_targets).sum(axis=1) / np.maximum(w_sum, 1e-12)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        R² score (coefficient of determination).

        Returns
        -------
        r2 : float
        """
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot == 0.0:
            raise ValueError("R² is undefined when y is constant.")
        return float(1.0 - ss_res / ss_tot)
