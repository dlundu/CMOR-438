"""Unit tests for rice_ml.processing.post_processing"""
import numpy as np
import pytest
from rice_ml.processing import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    mse,
    rmse,
    mae,
    r2_score,
)


# ---------------------- Classification: binary ----------------------

def test_accuracy_score():
    assert accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert accuracy_score([1, 1, 1], [1, 1, 1]) == 1.0

def test_precision_recall_f1_binary():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert precision_score(y_true, y_pred, average="binary") == 1.0
    assert recall_score(y_true, y_pred, average="binary") == 0.5
    assert np.isclose(f1_score(y_true, y_pred, average="binary"), 2/3)

def test_confusion_matrix_binary():
    cm = confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
    assert cm.tolist() == [[2, 0], [1, 1]]

def test_roc_auc_binary():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert round(roc_auc_score(y_true, scores), 2) == 0.75

def test_log_loss_binary():
    y = np.array([0, 1])
    probs = np.array([0.1, 0.9])
    ll = log_loss(y, probs)
    assert np.isclose(ll, -np.mean([np.log(0.9), np.log(0.9)]))

def test_roc_auc_invalid():
    with pytest.raises(ValueError):
        roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3])  # only one class

def test_log_loss_invalid_probs():
    with pytest.raises(ValueError):
        log_loss([0, 1], [1.5, 0.5])  # out of range


# ---------------------- Classification: multiclass ----------------------

def test_multiclass_accuracy():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    assert accuracy_score(y_true, y_pred) == 0.5

def test_multiclass_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 2, 1])
    assert precision_score(y_true, y_pred, average="macro") == 0.5
    assert recall_score(y_true, y_pred, average="macro") == 0.5

def test_multiclass_confusion_matrix_shape():
    cm = confusion_matrix([0, 1, 2, 2], [0, 2, 2, 1])
    assert cm.shape == (3, 3)


# ---------------------- Regression ----------------------

def test_mse():
    y_true = np.array([3., -0.5, 2., 7.])
    y_pred = np.array([2.5, 0.0, 2., 8.])
    assert np.isclose(mse(y_true, y_pred), 0.375)

def test_rmse():
    y_true = np.array([3., -0.5, 2., 7.])
    y_pred = np.array([2.5, 0.0, 2., 8.])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(0.375))

def test_mae():
    y_true = np.array([3., -0.5, 2., 7.])
    y_pred = np.array([2.5, 0.0, 2., 8.])
    assert np.isclose(mae(y_true, y_pred), 0.5)

def test_r2_score():
    y_true = np.array([3., -0.5, 2., 7.])
    y_pred = np.array([2.5, 0.0, 2., 8.])
    assert round(r2_score(y_true, y_pred), 4) == 0.9486

def test_r2_perfect():
    y = np.array([1., 2., 3.])
    assert r2_score(y, y) == 1.0

def test_regression_length_mismatch():
    with pytest.raises(ValueError):
        mse([1, 2], [1])
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])
