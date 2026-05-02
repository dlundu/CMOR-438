"""
rice_ml
=======
A from-scratch machine learning library built for CMOR 438 / INDE 577.

Quick start
-----------
>>> from rice_ml.supervised_learning import DecisionTreeClassifier, KNNClassifier
>>> from rice_ml.unsupervised_learning import KMeans, PCA
>>> from rice_ml.processing import standardize, train_test_split, accuracy_score
"""

# --- Subpackages ---
from . import supervised_learning
from . import unsupervised_learning
from . import processing

# --- Supervised ---
from .supervised_learning import (
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
from .unsupervised_learning import (
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
    "supervised_learning",
    "unsupervised_learning",
    "processing",
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
    "SVD",
    "PCA",
    "KMeans",
    "DBSCAN",
    "LabelPropagation",
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
