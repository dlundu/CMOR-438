"""
multilayer_perceptron.py
A from-scratch implementation of a Multilayer Perceptron (MLP) for binary
classification, trained with backpropagation and batch gradient descent.

Covered in: CMOR 438 / INDE 577 - Data Science & Machine Learning
Topic:       Supervised Learning > Neural Networks > MLP

Architecture
------------
    Input layer  →  Hidden layers (ReLU)  →  Output layer (Sigmoid)

Each hidden layer uses ReLU activation to introduce non-linearity. The
output layer uses a Sigmoid to produce a probability in [0, 1], and
Binary Cross-Entropy is minimised via backpropagation.

Forward pass (layer L):
    Z[L] = A[L-1] @ W[L]   (bias column appended to A[L-1])
    A[L] = relu(Z[L])       (hidden layers)
    A[out] = sigmoid(Z[out]) (output layer)

Backward pass (chain rule):
    δ_out = A_out - y                           (output delta)
    δ[L]  = (δ[L+1] @ W[L+1].T) * relu'(Z[L]) (hidden deltas)
    dW[L] = A[L-1].T @ δ[L] / n_samples        (gradient)

Weights are updated: W[L] ← W[L] - η * dW[L]

Usage Example
-------------
>>> from rice_ml.supervised.multilayer_perceptron import MLPBinaryClassifier
>>> model = MLPBinaryClassifier(hidden_layer_sizes=(64, 32), eta=0.01, epochs=500)
>>> model.fit(X_train, y_train)
>>> print(model.score(X_test, y_test))
"""

import numpy as np
import warnings
from typing import List, Optional, Tuple

def _relu(Z: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, Z). Applied element-wise."""
    return np.maximum(0.0, Z)


def _relu_grad(Z: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where Z > 0, else 0."""
    return (Z > 0).astype(float)


def _sigmoid(Z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-Z))."""
    Z_clipped = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z_clipped))

class MLPBinaryClassifier:
    """
    Multilayer Perceptron for binary classification (labels: 0 and 1).

    Trained with full-batch gradient descent and backpropagation.
    Bias units are handled by appending a column of ones to each activation
    matrix before each weight multiplication, keeping weight matrices compact.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        Number of neurons in each hidden layer. For example, (64, 32) gives
        two hidden layers with 64 and 32 neurons respectively.
    eta : float, default=0.01
        Learning rate for gradient descent weight updates.
    epochs : int, default=100
        Number of full forward/backward passes over the training data.
    random_state : int or None, default=None
        Seed for reproducible weight initialisation.
    verbose : int, default=0
        If > 0, print the training loss every `verbose` epochs.

    Attributes
    ----------
    weights_ : list of ndarray
        Trained weight matrices (including bias row) for each layer.
    cost_history_ : list of float
        Binary cross-entropy loss recorded after each epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from rice_ml.supervised.multilayer_perceptron import MLPBinaryClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 4))
    >>> y = (X[:, 0] - X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
    >>> model = MLPBinaryClassifier(hidden_layer_sizes=(16, 8), eta=0.05,
    ...                             epochs=200, random_state=0)
    >>> model.fit(X, y)
    >>> model.score(X, y) > 0.85
    True
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        eta: float = 0.01,
        epochs: int = 100,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

        self.weights_: List[np.ndarray] = []
        self.cost_history_: List[float] = []
        self._n_layers: int = 0

    @staticmethod
    def _with_bias(A: np.ndarray) -> np.ndarray:
        """Append a column of ones to activation matrix A for the bias unit."""
        return np.hstack([A, np.ones((A.shape[0], 1))])

    def _init_weights(self, n_features: int) -> None:
        """
        He-style weight initialisation for all layers.

        Each weight matrix W has shape (n_in + 1, n_out), where the extra
        row corresponds to the bias unit. Scale = sqrt(2 / n_in) works well
        with ReLU activations and helps avoid vanishing/exploding gradients.
        """
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [n_features, *self.hidden_layer_sizes, 1]
        self._n_layers = len(layer_sizes) - 1
        self.weights_ = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / n_in)
            W = rng.standard_normal((n_in + 1, n_out)) * scale
            self.weights_.append(W)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Full forward pass through the network.

        Returns
        -------
        A : list of ndarray
            Activations at each layer (A[0] = X, A[-1] = output probabilities).
        Z : list of ndarray
            Pre-activations at each layer (Z[0] is a placeholder).
        """
        A: List[np.ndarray] = [X]
        Z: List[np.ndarray] = [np.empty(0)]   # placeholder so indices align

        # Hidden layers — ReLU
        for l in range(self._n_layers - 1):
            Z_l = self._with_bias(A[-1]) @ self.weights_[l]
            A_l = _relu(Z_l)
            Z.append(Z_l)
            A.append(A_l)

        # Output layer — Sigmoid
        Z_out = self._with_bias(A[-1]) @ self.weights_[-1]
        A_out = _sigmoid(Z_out)
        Z.append(Z_out)
        A.append(A_out)

        return A, Z

    def _cross_entropy(self, A_out: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
        """Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))."""
        p = np.clip(A_out, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _backward(
        self, A: List[np.ndarray], Z: List[np.ndarray], y: np.ndarray
    ) -> List[np.ndarray]:
        """
        Backpropagation — compute gradients for every weight matrix.

        Returns
        -------
        dW : list of ndarray
            Gradient of loss w.r.t. each weight matrix, same order as
            self.weights_.
        """
        n = y.shape[0]
        dW: List[np.ndarray] = [np.empty(0)] * self._n_layers
        deltas: List[np.ndarray] = [np.empty(0)] * (self._n_layers + 1)

        # Output delta: d(BCE)/d(Z_out) = A_out - y  (sigmoid + BCE simplifies)
        deltas[-1] = A[-1] - y.reshape(-1, 1)
        dW[-1] = self._with_bias(A[-2]).T @ deltas[-1] / n

        # Propagate deltas back through hidden layers
        for l in range(self._n_layers - 1, 0, -1):
            # Strip the bias row from the weight matrix before propagating
            W_no_bias = self.weights_[l][:-1, :]
            error = deltas[l + 1] @ W_no_bias.T
            deltas[l] = error * _relu_grad(Z[l])
            dW[l - 1] = self._with_bias(A[l - 1]).T @ deltas[l] / n

        return dW

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBinaryClassifier":
        """
        Train the MLP on binary-labelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Should be standardised before calling fit.
        y : array-like of shape (n_samples,)
            Binary labels encoded as 0 and 1.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y row counts differ: {len(X)} vs {len(y)}")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Labels must be 0 or 1. Encode before calling fit().")

        self._init_weights(X.shape[1])
        self.cost_history_ = []

        for epoch in range(1, self.epochs + 1):
            A, Z = self._forward(X)
            cost = self._cross_entropy(A[-1], y)
            self.cost_history_.append(cost)

            dW = self._backward(A, Z, y)
            for l in range(self._n_layers):
                self.weights_[l] -= self.eta * dW[l]

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"Epoch {epoch:>{len(str(self.epochs))}}/{self.epochs}  "
                      f"loss: {cost:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicted probability of class 1 for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples,)
            Values in [0, 1].
        """
        if not self.weights_:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")
        X = np.asarray(X, dtype=float)
        A, _ = self._forward(X)
        return A[-1].ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (0 or 1) using a 0.5 decision threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Classification accuracy on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        accuracy : float in [0, 1]
        """
        y_true = np.asarray(y, dtype=float).ravel()
        return float(np.mean(self.predict(X) == y_true))
