"""
decision_tree_regressor.py
==========================
A from-scratch implementation of a Decision Tree Regressor using the CART
algorithm with variance reduction (MSE) as the splitting criterion.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Supervised Learning > Decision Trees & Ensembles

Usage Example
-------------
>>> from rice_ml.supervised_learning.decision_tree_regressor import DecisionTreeRegressor
>>> model = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)
>>> print(model.score(X_test, y_test))
"""

import numpy as np
from typing import Optional, Union, Tuple

def _mse(y: np.ndarray) -> float:
    """Mean squared error of a target array around its mean (i.e., variance)."""
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - np.mean(y)) ** 2))


def _weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Weighted MSE of two child splits."""
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * _mse(y_left) + (len(y_right) / n) * _mse(y_right)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ---------------------------------------------------------------------------
# Tree Node
# ---------------------------------------------------------------------------

class _Node:
    """
    Represents a single node in the decision tree.

    Leaf nodes store a prediction value; internal nodes store a split rule
    (feature index + threshold) and references to left/right children.
    """

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
        *,
        value: Optional[float] = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Only set for leaf nodes

    def is_leaf(self) -> bool:
        return self.value is not None

class DecisionTreeRegressor:
    """
    Decision Tree Regressor (CART) using variance reduction (MSE) as the
    splitting criterion.

    The algorithm greedily selects the feature and threshold that maximally
    reduces weighted MSE at each split, recursing until a stopping condition
    is met. Leaf nodes predict the mean of the target values they contain.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, nodes are expanded until all
        leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a split at a node.
    max_features : int, float, str, or None, default=None
        Number of features to consider when looking for the best split.
        - None  : use all features
        - int   : use exactly that many features
        - float : fraction of total features (e.g. 0.5)
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
    random_state : int or None, default=None
        Seed for the random number generator (used by max_features sampling).

    Attributes
    ----------
    root_ : _Node
        The root node of the fitted tree.
    n_features_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from rice_ml.supervised_learning.decision_tree_regressor import DecisionTreeRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 3))
    >>> y = 3 * X[:, 0] - 2 * X[:, 1] + rng.standard_normal(200) * 0.5
    >>> model = DecisionTreeRegressor(max_depth=4, random_state=0)
    >>> model.fit(X, y)
    >>> round(model.score(X, y), 2)
    0.98
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[_Node] = None
        self.n_features_: Optional[int] = None

    def _resolve_feature_indices(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        """Return the array of feature column indices to evaluate at a split."""
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError(f"Unsupported max_features string: '{self.max_features}'")
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            k = self.max_features
        else:
            raise TypeError(f"max_features must be int, float, str, or None — got {type(self.max_features)}")

        k = min(k, n_features)
        return rng.choice(n_features, size=k, replace=False)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Scan candidate splits across the chosen feature subset and return the
        one that gives the greatest reduction in weighted MSE.

        Returns
        -------
        best_feature : int or None
        best_threshold : float or None
        best_reduction : float   (0.0 if no valid split found)
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        parent_mse = _mse(y)
        feature_indices = self._resolve_feature_indices(n_features, rng)

        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_reduction = 0.0

        for feat in feature_indices:
            # Each unique value in the column is a candidate threshold
            for threshold in np.unique(X[:, feat]):
                mask = X[:, feat] <= threshold
                y_left, y_right = y[mask], y[~mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                reduction = parent_mse - _weighted_mse(y_left, y_right)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feat
                    best_threshold = threshold

        return best_feature, best_threshold, best_reduction

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> _Node:
        """Recursively build the tree, returning the root node of each subtree."""
        leaf_value = float(np.mean(y))

        # --- Stopping conditions ---
        if (
            _mse(y) < 1e-10                                      # node is already pure
            or len(y) < self.min_samples_split                   # too few samples
            or (self.max_depth is not None and depth >= self.max_depth)  # depth limit
        ):
            return _Node(value=leaf_value)

        feat, threshold, reduction = self._best_split(X, y, rng)

        # No beneficial split found → make a leaf
        if feat is None or reduction <= 0.0:
            return _Node(value=leaf_value)

        mask = X[:, feat] <= threshold
        left_child  = self._build(X[mask],  y[mask],  depth + 1, rng)
        right_child = self._build(X[~mask], y[~mask], depth + 1, rng)

        return _Node(feature_idx=feat, threshold=threshold, left=left_child, right=right_child)

    def _predict_one(self, x: np.ndarray, node: _Node) -> float:
        """Traverse the fitted tree to predict a single sample."""
        if node.is_leaf():
            return node.value  # type: ignore[return-value]
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """
        Build the regression tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of rows: {len(X)} vs {len(y)}")

        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if self.root_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        return np.array([self._predict_one(row, self.root_) for row in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R² score on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        r2 : float
        """
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return _r2_score(y_true, y_pred)

    def get_depth(self) -> int:
        """Return the maximum depth of the fitted tree."""
        if self.root_ is None:
            raise RuntimeError("Model is not fitted yet.")

        def _depth(node: Optional[_Node]) -> int:
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root_)

    def get_n_leaves(self) -> int:
        """Return the total number of leaf nodes in the fitted tree."""
        if self.root_ is None:
            raise RuntimeError("Model is not fitted yet.")

        def _count_leaves(node: Optional[_Node]) -> int:
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)

        return _count_leaves(self.root_)
