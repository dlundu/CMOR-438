"""
perceptron.py
=============
A from-scratch implementation of the Perceptron algorithm for binary
classification, based on Rosenblatt's original learning rule (1958).

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Supervised Learning > Neural Networks > Perceptron

How it works
------------
Each sample x is passed through a linear combination of weights and a bias
(the "net input"), then a step activation function maps the result to a
class label of +1 or -1:

    net_input(x) = w · x + b
    predict(x)   = +1  if net_input(x) >= 0
                   -1  otherwise

Weights are updated online (one sample at a time) using the rule:
    w ← w + η * (target - prediction) * x
    b ← b + η * (target - prediction)

If the data are linearly separable, the Perceptron Convergence Theorem
guarantees the algorithm will find a separating hyperplane in finite steps.

Usage Example
-------------
>>> from rice_ml.supervised.perceptron import Perceptron
>>> model = Perceptron(eta=0.01, epochs=100, random_state=42)
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)
>>> print(model.score(X_test, y_test))
"""

import numpy as np
from typing import Optional


class Perceptron:
    """
    Binary Perceptron classifier using the Rosenblatt learning rule.

    Labels must be encoded as +1 and -1. The model trains online — weights
    are updated after every misclassified sample — and stops early if a
    full epoch passes with zero errors (convergence).

    Parameters
    ----------
    eta : float, default=0.01
        Learning rate. Controls the step size of each weight update.
        Typical range: 0.001 – 0.5.
    epochs : int, default=50
        Maximum number of full passes over the training data.
    random_state : int or None, default=None
        Seed for weight initialization, ensuring reproducibility.

    Attributes
    ----------
    w_ : ndarray of shape (n_features,)
        Feature weights after fitting.
    b_ : float
        Bias term after fitting.
    errors_ : list of int
        Number of misclassifications recorded at the end of each epoch.
        Useful for plotting a convergence (learning) curve.
    n_epochs_trained_ : int
        Actual number of epochs run (may be less than `epochs` if the
        model converged early).

    Examples
    --------
    >>> import numpy as np
    >>> from rice_ml.supervised.perceptron import Perceptron
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 2))
    >>> y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)   # linearly separable
    >>> model = Perceptron(eta=0.1, epochs=100, random_state=0)
    >>> model.fit(X, y)
    >>> model.score(X, y)
    1.0
    """

    def __init__(
        self,
        eta: float = 0.01,
        epochs: int = 50,
        random_state: Optional[int] = None,
    ):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

        # Set after fit()
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.errors_: list[int] = []
        self.n_epochs_trained_: int = 0

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear pre-activation: z = X @ w + b.

        Works for a single sample (1-D) or a batch (2-D).
        """
        return np.dot(X, self.w_) + self.b_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the Perceptron on labelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels encoded as {-1, +1}.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If labels other than +1 / -1 are detected, or if X and y
            have incompatible shapes.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of rows: {len(X)} vs {len(y)}"
            )
        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be encoded as -1 and +1.")

        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.random(X.shape[1])   # small positive initial weights
        self.b_ = 0.0
        self.errors_ = []
        self.n_epochs_trained_ = 0

        for epoch in range(1, self.epochs + 1):
            epoch_errors = 0

            for xi, target in zip(X, y):
                prediction = 1.0 if self._net_input(xi) >= 0.0 else -1.0
                delta = self.eta * (target - prediction)

                if delta != 0.0:            # misclassified — apply update
                    self.w_ += delta * xi
                    self.b_ += delta
                    epoch_errors += 1

            self.errors_.append(epoch_errors)
            self.n_epochs_trained_ = epoch

            if epoch_errors == 0:           # converged — stop early
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels, each +1 or -1.
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        return np.where(self._net_input(X) >= 0.0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples (0.0 – 1.0).
        """
        y_true = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y_true == y_pred))

    def converged(self) -> bool:
        """
        Return True if training ended with zero misclassifications.

        A True result means the data were linearly separable and the
        Perceptron found a perfect separating hyperplane.
        """
        if not self.errors_:
            raise RuntimeError("Model is not fitted yet.")
        return self.errors_[-1] == 0
