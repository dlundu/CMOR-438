"""
post_processing.py
Evaluation metrics for classification and regression.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Data Processing & Fundamentals > Metrics

Classification
--------------
accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, roc_auc_score, log_loss

Regression
----------
mse, rmse, mae, r2_score

Usage Example
-------------
>>> from rice_ml.processing.postprocessing import accuracy_score, r2_score
>>> print(accuracy_score(y_true, y_pred))
>>> print(r2_score(y_true, y_pred))
"""

import numpy as np
from typing import Optional, Sequence

__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "roc_auc_score",
    "log_loss",
    "mse",
    "rmse",
    "mae",
    "r2_score",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _1d(arr, name):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr

def _1d_float(arr, name):
    arr = _1d(arr, name)
    try:
        return arr.astype(float)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must contain numeric values.")

def _check_len(yt, yp):
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred lengths differ: {len(yt)} vs {len(yp)}")

def _cm_raw(y_true, y_pred, labels=None):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    lmap = {v: i for i, v in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp):
        if t in lmap and p in lmap:
            cm[lmap[t], lmap[p]] += 1
    return cm, labels


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, *, labels=None) -> np.ndarray:
    """
    Confusion matrix: rows = true class, columns = predicted class.

    Parameters
    ----------
    y_true, y_pred : array-like (n_samples,)
    labels         : sequence, optional — class label order

    Returns
    -------
    cm : ndarray (n_classes, n_classes)
    """
    cm, _ = _cm_raw(y_true, y_pred, labels)
    return cm


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of correctly classified samples."""
    yt = _1d(y_true, "y_true")
    yp = _1d(y_pred, "y_pred")
    _check_len(yt, yp)
    return float(np.mean(yt == yp))


def precision_score(y_true, y_pred, *, average="binary", labels=None) -> float:
    """
    Precision: TP / (TP + FP).

    Parameters
    ----------
    average : 'binary' | 'macro' | 'micro'
    """
    yt = _1d(y_true, "y_true")
    yp = _1d(y_pred, "y_pred")
    cm, lbls = _cm_raw(yt, yp, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp

    if average == "binary":
        return float(tp[1] / (tp[1] + fp[1])) if (tp[1] + fp[1]) > 0 else 0.0
    per = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    if average == "macro":  return float(per.mean())
    if average == "micro":
        TP, FP = tp.sum(), fp.sum()
        return float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    raise ValueError("average must be 'binary', 'macro', or 'micro'.")


def recall_score(y_true, y_pred, *, average="binary", labels=None) -> float:
    """
    Recall: TP / (TP + FN).

    Parameters
    ----------
    average : 'binary' | 'macro' | 'micro'
    """
    yt = _1d(y_true, "y_true")
    yp = _1d(y_pred, "y_pred")
    cm, lbls = _cm_raw(yt, yp, labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp

    if average == "binary":
        return float(tp[1] / (tp[1] + fn[1])) if (tp[1] + fn[1]) > 0 else 0.0
    per = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    if average == "macro":  return float(per.mean())
    if average == "micro":
        TP, FN = tp.sum(), fn.sum()
        return float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    raise ValueError("average must be 'binary', 'macro', or 'micro'.")


def f1_score(y_true, y_pred, *, average="binary", labels=None) -> float:
    """Harmonic mean of precision and recall."""
    p = precision_score(y_true, y_pred, average=average, labels=labels)
    r = recall_score(y_true, y_pred, average=average, labels=labels)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def roc_auc_score(y_true, y_scores) -> float:
    """
    ROC AUC for binary classification (rank-based, exact).

    Parameters
    ----------
    y_true   : array-like — true binary labels
    y_scores : array-like — scores for the positive class
    """
    yt = _1d(y_true, "y_true")
    ys = _1d_float(y_scores, "y_scores")
    _check_len(yt, ys)
    uniq = np.unique(yt)
    if uniq.size != 2:
        raise ValueError("roc_auc_score requires exactly 2 classes.")
    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ys) + 1)
    pos = yt == uniq[1]
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(y_true, y_prob, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy loss.

    Parameters
    ----------
    y_true  : array-like — true binary labels (0/1)
    y_prob  : array-like — predicted probabilities for class 1
    """
    yt = _1d_float(y_true, "y_true")
    yp = _1d_float(y_prob, "y_prob")
    _check_len(yt, yp)
    if np.any((yp < 0) | (yp > 1)):
        raise ValueError("y_prob must be in [0, 1].")
    p = np.clip(yp, eps, 1 - eps)
    return float(-np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mse(y_true, y_pred) -> float:
    """Mean squared error."""
    yt, yp = _1d_float(y_true, "y_true"), _1d_float(y_pred, "y_pred")
    _check_len(yt, yp)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    yt, yp = _1d_float(y_true, "y_true"), _1d_float(y_pred, "y_pred")
    _check_len(yt, yp)
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true, y_pred) -> float:
    """
    Coefficient of determination R².

    Returns 1.0 if predictions are perfect on a constant target,
    raises ValueError if target is constant and predictions are not.
    """
    yt, yp = _1d_float(y_true, "y_true"), _1d_float(y_pred, "y_pred")
    _check_len(yt, yp)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)
