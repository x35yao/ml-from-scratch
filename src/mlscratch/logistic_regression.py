import numpy as np
from mlscratch.utils import standardize, ensure_2D, sigmoid


class LogisticRegression:
    def __init__(self):
        self.w = None                # stacked [coef, intercept]
        self.coef_ = None
        self.intercept_ = None
        self.x_mean = None
        self.x_std = None


    def fit(self, X, y, lr = 1e-2, max_iter = 1000, tol = 1e-8, random_state = 0):

        X = ensure_2D(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if not np.array_equal(np.unique(y), [0, 1]) and not np.array_equal(np.unique(y), [0]) and not np.array_equal(np.unique(y), [1]):
            raise ValueError("y must be binary with values {0,1}.")
        N, D = X.shape[0], X.shape[1]

        rng = np.random.default_rng(random_state)
        w = rng.normal(scale = 1e-3, size = D)
        b = 0.0

        Xn, self.x_mean, self.x_std = standardize(X)

        for it in range(max_iter):
            z = Xn @ w + b
            p = sigmoid(z)                 # shape (N,)

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
        X = ensure_2D(X)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        z = Xb @ self.w
        return sigmoid(z)  # P(y=1)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)

    def score(self, X, y):
        y = np.asarray(y).ravel()
        return np.mean(y == self.predict(X))
