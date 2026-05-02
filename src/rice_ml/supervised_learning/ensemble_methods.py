"""
ensemble_methods.py
Ensemble learning methods for classification and regression.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Supervised Learning > Decision Trees & Ensembles

Classes
-------
- VotingClassifier         : Hard majority vote across diverse base classifiers.
- BaggingClassifier        : Bootstrap aggregating over a base learner factory.
- RandomForestClassifier   : Bagging + random feature subsets per split.
- AdaBoostClassifier       : Sequential re-weighting of misclassified samples.
- GradientBoostingRegressor: Sequential residual fitting (MSE loss).

Usage Example
-------------
>>> from rice_ml.supervised.ensemble import RandomForestClassifier
>>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
>>> rf.fit(X_train, y_train)
>>> rf.score(X_test, y_test)
"""

import numpy as np
from typing import Callable, List, Optional, Union
from .regression_trees import DecisionTreeRegressor
def _validate(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y is None:
        return X
    y = np.asarray(y).ravel()
    if len(X) != len(y):
        raise ValueError(f"X and y lengths differ: {len(X)} vs {len(y)}")
    return X, y


def _majority_vote(votes: np.ndarray) -> np.ndarray:
    """Column-wise majority vote over a (n_estimators, n_samples) matrix."""
    result = np.empty(votes.shape[1], dtype=votes.dtype)
    for j in range(votes.shape[1]):
        labels, counts = np.unique(votes[:, j], return_counts=True)
        result[j] = labels[np.argmax(counts)]
    return result

class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature, self.threshold = feature, threshold
        self.left, self.right = left, right
        self.value = value

    def is_leaf(self): return self.value is not None


class _DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None

    @staticmethod
    def _gini(y):
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        return 1.0 - float(np.sum(p ** 2))

    def _feat_idx(self, n, rng):
        mf = self.max_features
        if mf is None: return np.arange(n)
        k = (max(1, int(np.sqrt(n))) if mf == "sqrt" else
             max(1, int(np.log2(n))) if mf == "log2" else
             max(1, int(mf * n)) if isinstance(mf, float) else int(mf))
        return rng.choice(n, size=min(k, n), replace=False)

    def _split(self, X, y, rng):
        best_gain, feat, thresh = 0.0, None, None
        parent = self._gini(y)
        n = len(y)
        for f in self._feat_idx(X.shape[1], rng):
            for t in np.unique(X[:, f]):
                m = X[:, f] <= t
                yl, yr = y[m], y[~m]
                if not len(yl) or not len(yr): continue
                gain = parent - (len(yl)/n * self._gini(yl) + len(yr)/n * self._gini(yr))
                if gain > best_gain:
                    best_gain, feat, thresh = gain, f, t
        return feat, thresh, best_gain

    def _build(self, X, y, depth, rng):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or \
                (self.max_depth and depth >= self.max_depth):
            return _Node(value=int(np.bincount(y).argmax()))
        f, t, gain = self._split(X, y, rng)
        if f is None or gain <= 0:
            return _Node(value=int(np.bincount(y).argmax()))
        m = X[:, f] <= t
        return _Node(f, t, self._build(X[m], y[m], depth+1, rng),
                          self._build(X[~m], y[~m], depth+1, rng))

    def fit(self, X, y):
        X, y = _validate(X, y)
        self.root_ = self._build(X, y.astype(int), 0, np.random.default_rng(self.random_state))
        return self

    def _traverse(self, x, node):
        if node.is_leaf(): return node.value
        return self._traverse(x, node.left if x[node.feature] <= node.threshold else node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(row, self.root_) for row in X])

class VotingClassifier:
    """
    Hard-voting ensemble. All estimators are trained on the full dataset and
    each gets one vote per prediction.

    Parameters
    ----------
    estimators : list of classifiers with fit/predict API.
    """
    def __init__(self, estimators: List):
        self.estimators = estimators

    def fit(self, X, y):
        X, y = _validate(X, y)
        for e in self.estimators: e.fit(X, y)
        return self

    def predict(self, X):
        X = _validate(X)
        return _majority_vote(np.array([e.predict(X) for e in self.estimators]))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

class BaggingClassifier:
    """
    Bootstrap Aggregating for classification.

    Parameters
    ----------
    base_learner : callable returning a new unfitted classifier, default=_DecisionTreeClassifier
    n_estimators : int, default=10
    max_samples  : float, fraction of training set per bootstrap draw, default=1.0
    random_state : int or None
    """
    def __init__(self, base_learner=None, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_learner = base_learner or (lambda: _DecisionTreeClassifier())
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_: List = []

    def fit(self, X, y):
        X, y = _validate(X, y)
        n, k = len(X), max(1, int(self.max_samples * len(X)))
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=k, replace=True)
            e = self.base_learner()
            e.fit(X[idx], y[idx])
            self.estimators_.append(e)
        return self

    def predict(self, X):
        X = _validate(X)
        return _majority_vote(np.array([e.predict(X) for e in self.estimators_]))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

class RandomForestClassifier:
    """
    Bagging of decision trees with random feature subsets per split.

    Parameters
    ----------
    n_estimators     : int, default=100
    max_depth        : int or None
    max_features     : 'sqrt', 'log2', float, int, or None — default='sqrt'
    min_samples_split: int, default=2
    max_samples      : float, default=1.0
    random_state     : int or None

    Attributes
    ----------
    feature_importances_ : ndarray — split-frequency importance per feature.
    """
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt",
                 min_samples_split=2, max_samples=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_: List[_DecisionTreeClassifier] = []
        self.feature_importances_ = None

    def _count_splits(self, node, counts):
        if node is None or node.is_leaf(): return
        counts[node.feature] += 1
        self._count_splits(node.left, counts)
        self._count_splits(node.right, counts)

    def fit(self, X, y):
        X, y = _validate(X, y)
        n, n_feat = X.shape
        k = max(1, int(self.max_samples * n))
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = []
        for seed in seeds:
            idx = np.random.default_rng(seed).choice(n, size=k, replace=True)
            tree = _DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                max_features=self.max_features, random_state=int(seed))
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        raw = np.zeros(n_feat)
        for t in self.estimators_: self._count_splits(t.root_, raw)
        self.feature_importances_ = raw / raw.sum() if raw.sum() > 0 else raw
        return self

    def predict(self, X):
        X = _validate(X)
        return _majority_vote(np.array([t.predict(X) for t in self.estimators_]))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
class AdaBoostClassifier:
    """
    AdaBoost for binary classification using decision stumps (max_depth=1).

    Parameters
    ----------
    n_estimators  : int, default=50
    learning_rate : float, default=1.0
    random_state  : int or None
    """
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_: List = []
        self.alphas_: List[float] = []
        self.classes_ = None

    def fit(self, X, y):
        X, y = _validate(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostClassifier supports only binary classification.")
        y_s = np.where(y == self.classes_[1], 1.0, -1.0)
        w = np.full(len(X), 1.0 / len(X))
        rng = np.random.default_rng(self.random_state)
        self.estimators_, self.alphas_ = [], []
        for seed in rng.integers(0, 2**31, size=self.n_estimators):
            stump = _DecisionTreeClassifier(max_depth=1, random_state=int(seed))
            stump.fit(X, y)
            pred_s = np.where(stump.predict(X) == self.classes_[1], 1.0, -1.0)
            eps = np.clip(np.dot(w, pred_s != y_s) / w.sum(), 1e-10, 1 - 1e-10)
            alpha = self.learning_rate * 0.5 * np.log((1 - eps) / eps)
            w *= np.exp(-alpha * y_s * pred_s)
            w /= w.sum()
            self.estimators_.append(stump)
            self.alphas_.append(alpha)
        return self

    def predict(self, X):
        X = _validate(X)
        weighted = sum(a * np.where(e.predict(X) == self.classes_[1], 1.0, -1.0)
                       for e, a in zip(self.estimators_, self.alphas_))
        return np.where(weighted >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

class GradientBoostingRegressor:
    """
    Gradient Boosting for regression (MSE loss).

    Sequentially fits trees to the residuals of the current ensemble:
        F_0(x) = mean(y)
        F_m(x) = F_{m-1}(x) + learning_rate * tree_m(residuals)

    Parameters
    ----------
    n_estimators      : int, default=100
    learning_rate     : float, default=0.1
    max_depth         : int, default=3
    min_samples_split : int, default=2
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators_: List[DecisionTreeRegressor] = []
        self.init_: float = 0.0

    def fit(self, X, y):
        X, y = _validate(X, y)
        y = y.astype(float)
        self.init_ = float(np.mean(y))
        F = np.full(len(y), self.init_)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split)
            tree.fit(X, y - F)
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X = _validate(X)
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X, y):
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
