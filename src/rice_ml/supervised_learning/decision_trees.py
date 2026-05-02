"""
decision_tree_classifier.py
============================
Decision Tree Classifier (CART) using Gini impurity or Entropy.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Supervised Learning > Decision Trees & Ensembles

Usage Example
-------------
>>> from rice_ml.supervised.decision_tree_classifier import DecisionTreeClassifier
>>> clf = DecisionTreeClassifier(max_depth=4, random_state=42)
>>> clf.fit(X_train, y_train)
>>> clf.score(X_test, y_test)
"""

import numpy as np
from typing import Literal, Optional, Union

class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature, self.threshold = feature, threshold
        self.left, self.right = left, right
        self.value = value

    def is_leaf(self): return self.value is not None


def _gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1.0 - float(np.sum(p ** 2))


def _entropy(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


class DecisionTreeClassifier:
    """
    Decision Tree Classifier (CART) with Gini or Entropy splitting criterion.

    Parameters
    ----------
    criterion        : 'gini' or 'entropy', default='gini'
    max_depth        : int or None
    min_samples_split: int, default=2
    max_features     : 'sqrt', 'log2', float, int, or None
    random_state     : int or None
    """

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[Union[str, int, float]] = None,
        random_state: Optional[int] = None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got '{criterion}'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._impurity = _gini if criterion == "gini" else _entropy
        self.root_: Optional[_Node] = None
        self.n_features_: Optional[int] = None

    def _feat_indices(self, n, rng):
        mf = self.max_features
        if mf is None: return np.arange(n)
        k = (max(1, int(np.sqrt(n))) if mf == "sqrt" else
             max(1, int(np.log2(n))) if mf == "log2" else
             max(1, int(mf * n)) if isinstance(mf, float) else int(mf))
        return rng.choice(n, size=min(k, n), replace=False)

    def _best_split(self, X, y, rng):
        n = len(y)
        if n < self.min_samples_split:
            return None, None, -1.0
        parent = self._impurity(y)
        best_gain, feat, thresh = -1.0, None, None
        for f in self._feat_indices(X.shape[1], rng):
            for t in np.unique(X[:, f]):
                m = X[:, f] <= t
                yl, yr = y[m], y[~m]
                if not len(yl) or not len(yr): continue
                gain = parent - (len(yl)/n * self._impurity(yl) + len(yr)/n * self._impurity(yr))
                if gain > best_gain:
                    best_gain, feat, thresh = gain, f, t
        return feat, thresh, best_gain

    def _build(self, X, y, depth, rng):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or \
                (self.max_depth is not None and depth >= self.max_depth):
            return _Node(value=self._leaf_value(y))
        f, t, gain = self._best_split(X, y, rng)
        if f is None or gain <= 0:
            return _Node(value=self._leaf_value(y))
        m = X[:, f] <= t
        return _Node(f, t, self._build(X[m], y[m], depth+1, rng),
                          self._build(X[~m], y[~m], depth+1, rng))

    @staticmethod
    def _leaf_value(y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _traverse(self, x, node):
        if node.is_leaf(): return node.value
        return self._traverse(x, node.left if x[node.feature] <= node.threshold else node.right)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2: raise ValueError(f"X must be 2-D, got {X.shape}")
        if len(X) != len(y): raise ValueError("X and y lengths differ.")
        self.n_features_ = X.shape[1]
        self.root_ = self._build(X, y, 0, np.random.default_rng(self.random_state))
        return self

    def predict(self, X):
        if self.root_ is None: raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(row, self.root_) for row in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
