"""Unit tests for rice_ml.processing.pre_processing"""
import numpy as np
import pytest
from rice_ml.processing import (
    standardize,
    minmax_scale,
    train_test_split,
    train_val_test_split,
)


# ---------------------- standardize ----------------------

def test_standardize_zero_mean():
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    Z = standardize(X)
    assert np.allclose(Z.mean(axis=0), 0.0)

def test_standardize_unit_std():
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    Z = standardize(X)
    assert np.allclose(Z.std(axis=0), 1.0)

def test_standardize_constant_column():
    X = np.array([[1., 5.], [3., 5.], [5., 5.]])
    Z = standardize(X)
    # constant column should not blow up — stays 0
    assert np.allclose(Z[:, 1], 0.0)

def test_standardize_no_mean():
    X = np.array([[1., 2.], [3., 4.]])
    Z = standardize(X, with_mean=False)
    assert Z.shape == X.shape

def test_standardize_no_std():
    X = np.array([[1., 2.], [3., 4.]])
    Z = standardize(X, with_std=False)
    assert np.allclose(Z.mean(axis=0), 0.0)

def test_standardize_invalid_input():
    with pytest.raises(ValueError):
        standardize(np.array([1., 2., 3.]))  # 1D not allowed


# ---------------------- minmax_scale ----------------------

def test_minmax_scale_range():
    X = np.array([[0., 0.], [5., 10.], [10., 10.]])
    Z = minmax_scale(X)
    assert np.allclose(Z.min(axis=0), 0.0)
    assert np.allclose(Z.max(axis=0)[0], 1.0)

def test_minmax_scale_custom_range():
    X = np.array([[0.], [5.], [10.]])
    Z = minmax_scale(X, feature_range=(-1, 1))
    assert np.isclose(Z.min(), -1.0)
    assert np.isclose(Z.max(), 1.0)

def test_minmax_scale_constant_column():
    X = np.array([[1., 3.], [2., 3.], [3., 3.]])
    Z = minmax_scale(X)
    # constant column should not blow up
    assert np.all(np.isfinite(Z))

def test_minmax_scale_invalid_range():
    X = np.array([[1., 2.], [3., 4.]])
    with pytest.raises(ValueError):
        minmax_scale(X, feature_range=(1, 1))

def test_minmax_scale_invalid_input():
    with pytest.raises(ValueError):
        minmax_scale(np.array([1., 2., 3.]))  # 1D not allowed


# ---------------------- train_test_split ----------------------

def test_train_test_split_sizes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    assert len(X_tr) + len(X_te) == 50
    assert len(y_tr) + len(y_te) == 50

def test_train_test_split_reproducible():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.2, random_state=42)
    assert np.array_equal(X_tr1, X_tr2)
    assert np.array_equal(y_te1, y_te2)

def test_train_test_split_no_shuffle():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    assert len(X_tr) == 8 and len(X_te) == 2

def test_train_test_split_without_y():
    X = np.arange(20).reshape(10, 2)
    X_tr, X_te = train_test_split(X, test_size=0.3)
    assert len(X_tr) + len(X_te) == 10

def test_train_test_split_invalid():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)


# ---------------------- train_val_test_split ----------------------

def test_train_val_test_split_sizes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        X, y, val_size=0.1, test_size=0.2, random_state=0
    )
    assert len(X_tr) + len(X_va) + len(X_te) == 50

def test_train_val_test_split_without_y():
    X = np.arange(30).reshape(15, 2)
    X_tr, X_va, X_te = train_val_test_split(X, val_size=0.2, test_size=0.2)
    assert len(X_tr) + len(X_va) + len(X_te) == 15

def test_train_val_test_split_invalid():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    with pytest.raises(ValueError):
        train_val_test_split(X, y, val_size=0.6, test_size=0.5)
