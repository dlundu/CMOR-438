import numpy as np
from typing import Optional, List

class LogisticRegression:
    """
    Logistic Regression Classifier using Batch Gradient Descent.
    Custom implementation for binary classification.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, random_state: Optional[int] = None):
        self.lr = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights_: Optional[np.ndarray] = None
        self.cost_history_: List[float] = []

    def _sigmoid(self, z):
        # Internal helper for sigmoid activation
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _add_bias(self, X):
        # Internal helper to add the intercept column
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def _compute_cost(self, y, y_pred, eps=1e-15):
        # Binary Cross-Entropy Loss
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        
        n_samples, n_features = X_arr.shape
        rng = np.random.default_rng(self.random_state)
        
        # Initialize weights (including bias)
        self.weights_ = rng.standard_normal(size=n_features + 1) * 0.01
        self.cost_history_ = []

        X_biased = self._add_bias(X_arr)
        
        for _ in range(self.epochs):
            # Forward pass
            net_input = X_biased @ self.weights_
            y_pred = self._sigmoid(net_input)

            # Gradient calculation
            error = y_pred - y_arr
            gradient = (X_biased.T @ error) / n_samples
            
            # Update weights
            self.weights_ -= self.lr * gradient
            
            # Record cost
            self.cost_history_.append(self._compute_cost(y_arr, y_pred))
            
        return self

    def predict_proba(self, X):
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X_biased = self._add_bias(np.asarray(X))
        return self._sigmoid(X_biased @ self.weights_)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
