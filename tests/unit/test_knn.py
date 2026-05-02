"""Unit tests for rice_ml.supervised_learning.knn"""
import numpy as np
import pytest
from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


# ---------------------- KNNClassifier ----------------------

def test_classifier_basic_predict():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=3).fit(X, y)
    preds = clf.predict([[0.1, 0.1], [0.9, 0.9]])
    assert preds.tolist() == [0, 1]

def test_classifier_predict_proba_sums_to_one():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=3).fit(X, y)
    proba = clf.predict_proba([[0.1, 0.1], [0.9, 0.9]])
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_classifier_perfect_score():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=1).fit(X, y)
    assert clf.score(X, y) == 1.0

def test_classifier_manhattan_metric():
    X = np.array([[0,0],[2,0],[0,2],[2,2]], dtype=float)
    y = np.array([0, 0, 1, 1])
    clf = KNNClassifier(n_neighbors=3, metric="manhattan").fit(X, y)
    pred = clf.predict([[0.1, 0.1]])
    assert pred[0] == 0

def test_classifier_distance_weights():
    X = np.array([[0,0],[1,1],[0,0]], dtype=float)
    y = np.array([0, 1, 0])
    clf = KNNClassifier(n_neighbors=2, weights="distance").fit(X, y)
    pred = clf.predict([[0, 0]])
    assert pred[0] == 0

def test_classifier_kneighbors_shape():
    X = np.array([[0,0],[1,1],[2,2]], dtype=float)
    y = np.array([0, 1, 1])
    clf = KNNClassifier(n_neighbors=2).fit(X, y)
    d, idx = clf.kneighbors([[1.0, 1.0]])
    assert d.shape == (1, 2) and idx.shape == (1, 2)

def test_classifier_wrong_feature_count():
    X = np.array([[0,0],[1,1],[2,2]], dtype=float)
    y = np.array([0, 1, 1])
    clf = KNNClassifier(n_neighbors=2).fit(X, y)
    with pytest.raises(ValueError):
        clf.predict([[0.0, 0.0, 0.0]])

def test_classifier_not_fitted():
    clf = KNNClassifier(n_neighbors=2)
    with pytest.raises(RuntimeError):
        clf.predict([[0.0, 0.0]])

def test_classifier_invalid_n_neighbors():
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=0)

def test_classifier_invalid_metric():
    with pytest.raises(ValueError):
        KNNClassifier(metric="cosine")


# ---------------------- KNNRegressor ----------------------

def test_regressor_basic_predict():
    X = np.array([[0],[1],[2],[3]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    reg = KNNRegressor(n_neighbors=1).fit(X, y)
    assert reg.score(X, y) == 1.0

def test_regressor_distance_weights():
    X = np.array([[0],[1],[2],[3]], dtype=float)
    y = np.array([0.0, 1.0, 1.5, 3.0])
    reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
    pred = reg.predict([[1.5]])[0]
    assert 1.0 < pred < 2.0

def test_regressor_n_neighbors_too_large():
    X = np.array([[0],[1],[2]], dtype=float)
    y = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        KNNRegressor(n_neighbors=5).fit(X, y)

def test_regressor_not_fitted():
    reg = KNNRegressor(n_neighbors=2)
    with pytest.raises(RuntimeError):
        reg.predict([[0.0]])

def test_regressor_r2_constant_y():
    X = np.array([[0],[1],[2]], dtype=float)
    y = np.array([5.0, 5.0, 5.0])
    reg = KNNRegressor(n_neighbors=1).fit(X, y)
    # perfect predictions on training -> R2 = 1.0
    assert reg.score(X, y) == 1.0
