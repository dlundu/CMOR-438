import numpy as np
from typing import Optional, Literal

class LinearModel:
    """
    Unified Linear Model: Supports OLS, Ridge, and Lasso.
    Calculates weights using closed-form solutions or coordinate descent.
    """
    def __init__(self, method: Literal['ols', 'ridge', 'lasso'] = 'ols', 
                 alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.intercept = 0.0

    def _add_bias(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def _soft_threshold(self, x, lam):
        """Helper for Lasso coordinate descent."""
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        if self.method == 'ols' or self.method == 'ridge':
            # Closed-form solution
            X_b = self._add_bias(X)
            # Normal Equation: (X^T X + alpha*I)
            XTX = X_b.T @ X_b
            if self.method == 'ridge':
                I = np.eye(XTX.shape[0])
                I[0, 0] = 0 # Don't penalize bias
                XTX += self.alpha * I
            
            theta = np.linalg.pinv(XTX) @ X_b.T @ y
            self.intercept = theta[0]
            self.weights = theta[1:]

        elif self.method == 'lasso':
            # Coordinate Descent for Lasso
            self.weights = np.zeros(n_features)
            self.intercept = np.mean(y)
            y_centered = y - self.intercept
            
            for _ in range(self.max_iter):
                weights_old = self.weights.copy()
                for j in range(n_features):
                    # Partial residual
                    prediction = X @ self.weights
                    residual = y_centered - (prediction - X[:, j] * self.weights[j])
                    
                    # Update feature weight
                    rho_j = X[:, j] @ residual
                    self.weights[j] = self._soft_threshold(rho_j, self.alpha) / (X[:, j] @ X[:, j])
                
                if np.linalg.norm(self.weights - weights_old) < self.tol:
                    break
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

    def score(self, X, y):
        """Returns R^2 score."""
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
