import numpy as np

class LogisticRegression:
    def __init__(self):
        self.w = None                # stacked [coef, intercept]
        self.coef_ = None
        self.intercept_ = None
        self.x_mean = None
        self.x_std = None
    
    def _sigmoid(self, z):
        # numerically stable sigmoid
        z = np.clip(z, -709, 709)    # exp overflow guard for float64
        return 1.0 / (1.0 + np.exp(-z))

    def _ensure_2D(self, X):
        """Ensure X is 2-D.

        Parameters
        ----------
        X : array_like of shape (N, D) or (N,)
            Input features.

        Returns
        -------

        X : ndarray of shape (N, D)
            If a 1-D array is passed, it is reshaped to (N, 1).

        Raises
        ------
        ValueError
            If X has more than 2 dimensions.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("X must be 1D or 2D")
        return X

    def fit(self, X, y, lr = 1e-2, max_iter = 1000, tol = 1e-8, random_state = 0):

        X = self._ensure_2D(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if not np.array_equal(np.unique(y), [0, 1]) and not np.array_equal(np.unique(y), [0]) and not np.array_equal(np.unique(y), [1]):
            raise ValueError("y must be binary with values {0,1}.")
        N, D = X.shape[0], X.shape[1]

        rng = np.random.default_rng(random_state)
        w = rng.normal(scale = 1e-3, size = D)
        b = 0.0

        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std  = X.std(axis=0, keepdims=True) + 1e-12
        Xn = (X - self.x_mean) / self.x_std

        for it in range(max_iter):
            z = Xn @ w + b
            p = self._sigmoid(z)                 # shape (N,)

            # gradient of average binary cross-entropy
            err = (p - y)                        # (N,)
            grad_w = Xn.T @ err / N              # (D,)
            grad_b = err.mean()                  # scalar

            step_w = lr * grad_w
            step_b = lr * grad_b

            w -= step_w
            b -=  step_b

            if np.linalg.norm(np.r_[step_w, step_b]) < tol:
                break

        # Map (w,b) back to original feature space and save
        coef = (w / self.x_std.ravel())  # (D,)
        intercept = float(b - (self.x_mean.ravel() / self.x_std.ravel()) @ w)

        # 5) Save params
        self.coef_ = coef
        self.intercept_ = intercept
        self.w = np.r_[coef, intercept]
        return self       
        

    def predict_proba(self, X):
        if self.w is None:
            raise ValueError("Model is not trained.")
        X = self._ensure_2D(X)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        z = Xb @ self.w
        return self._sigmoid(z)  # P(y=1)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)

    def score(self, X, y):
        y = np.asarray(y).ravel()
        return np.mean(y == self.predict(X))
