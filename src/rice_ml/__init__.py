"""
rice_ml
=======
A from-scratch machine learning library built for CMOR 438 / INDE 577.

Subpackages
-----------
supervised   : classification and regression models
unsupervised : clustering and dimensionality reduction
processing   : preprocessing utilities and evaluation metrics

Quick start
-----------
>>> from rice_ml.supervised import DecisionTreeClassifier, KNNClassifier
>>> from rice_ml.unsupervised import KMeans, PCA
>>> from rice_ml.processing import standardize, train_test_split, accuracy_score
"""

# --- Subpackages ---
from . import supervised
from . import unsupervised
from . import processing

# --- Supervised ---
from .supervised import (
    LinearModel,
    LogisticRegression,
    Perceptron,
    MLPBinaryClassifier,
    KNNClassifier,
    KNNRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingRegressor,
)

# --- Unsupervised ---
from .unsupervised import (
    SVD,
    PCA,
    KMeans,
    DBSCAN,
    LabelPropagation,
)

# --- Processing ---
from .processing import (
    standardize,
    minmax_scale,
    train_test_split,
    train_val_test_split,
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

__all__ = [
    # Subpackages
    "supervised",
    "unsupervised",
    "processing",
    # Supervised
    "LinearModel",
    "LogisticRegression",
    "Perceptron",
    "MLPBinaryClassifier",
    "KNNClassifier",
    "KNNRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "VotingClassifier",
    "BaggingClassifier",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "GradientBoostingRegressor",
    # Unsupervised
    "SVD",
    "PCA",
    "KMeans",
    "DBSCAN",
    "LabelPropagation",
    # Processing
    "standardize",
    "minmax_scale",
    "train_test_split",
    "train_val_test_split",
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
