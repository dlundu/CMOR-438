"""
pre_processing.py
Common data preprocessing utilities for machine learning.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Data Processing & Fundamentals

Functions
---------
standardize        : Zero-mean, unit-variance scaling (Z-score).
minmax_scale       : Scale features to a given range (default [0, 1]).
train_test_split   : Split arrays into train and test subsets.
train_val_test_split: Split arrays into train, validation, and test subsets.

Usage Example
-------------
>>> from rice_ml.processing.preprocessing import standardize, train_test_split
>>> X_scaled = standardize(X)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""

import numpy as np
from typing import Optional, Tuple, Union

__all__ = [
    "standardize",
    "minmax_scale",
    "train_test_split",
    "train_val_test_split",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float2d(X, name="X"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {X.shape}")
    if X.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return X

def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def standardize(
    X: np.ndarray,
    *,
    with_mean: bool = True,
    with_std: bool = True,
) -> np.ndarray:
    """
    Z-score standardization: (X - mean) / std, feature-wise.

    Columns with zero variance are left unchanged (divided by 1).

    Parameters
    ----------
    X         : array-like (n_samples, n_features)
    with_mean : subtract column mean, default=True
    with_std  : divide by column std, default=True

    Returns
    -------
    X_out : ndarray (n_samples, n_features)
    """
    X = _to_float2d(X)
    mean  = X.mean(axis=0) if with_mean else np.zeros(X.shape[1])
    X_out = X - mean
    if with_std:
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X_out /= std
    return X_out


def minmax_scale(
    X: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Scale features to `feature_range` (default [0, 1]).

    Constant features (max == min) are mapped to the lower bound.

    Parameters
    ----------
    X             : array-like (n_samples, n_features)
    feature_range : (min, max) output range

    Returns
    -------
    X_out : ndarray (n_samples, n_features)
    """
    X = _to_float2d(X)
    lo, hi = float(feature_range[0]), float(feature_range[1])
    if lo >= hi:
        raise ValueError("feature_range must have min < max.")
    X_min = X.min(axis=0)
    denom = X.max(axis=0) - X_min
    denom[denom == 0] = 1.0
    return (X - X_min) / denom * (hi - lo) + lo


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def train_test_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
):
    """
    Split arrays into train and test subsets.

    Parameters
    ----------
    X            : array-like (n_samples, n_features)
    y            : array-like (n_samples,), optional
    test_size    : fraction for test set, default=0.2
    shuffle      : shuffle before splitting, default=True
    random_state : int or None

    Returns
    -------
    X_train, X_test            if y is None
    X_train, X_test, y_train, y_test  otherwise
    """
    X = np.asarray(X)
    n = len(X)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1).")

    idx = _rng(random_state).permutation(n) if shuffle else np.arange(n)
    n_test  = max(1, int(round(test_size * n)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    if y is None:
        return X[train_idx], X[test_idx]
    y = np.asarray(y).ravel()
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_val_test_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    val_size: float = 0.1,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
):
    """
    Split arrays into train, validation, and test subsets.

    Parameters
    ----------
    X, y         : arrays (see train_test_split)
    val_size     : fraction for validation set, default=0.1
    test_size    : fraction for test set, default=0.2
    shuffle      : default=True
    random_state : int or None

    Returns
    -------
    X_train, X_val, X_test                       if y is None
    X_train, X_val, X_test, y_train, y_val, y_test  otherwise
    """
    if not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.")

    X = np.asarray(X)
    n = len(X)
    idx = _rng(random_state).permutation(n) if shuffle else np.arange(n)

    n_test  = max(1, int(round(test_size * n)))
    n_val   = max(1, int(round(val_size  * n)))
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    if y is None:
        return X[train_idx], X[val_idx], X[test_idx]
    y = np.asarray(y).ravel()
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])
