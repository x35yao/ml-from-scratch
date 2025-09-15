import numpy as np
class LinearRegression:
    def __init__(self):
        self.N = None
        self.D = None
        self.w = None # augmented-weight vector of shape (D + 1,)
        self.coef_ = None # shape (D, )
        self. intercept_ = None # float
        self.lr = 1e-3 # learning rate
        self.max_iter = 200
        self.tol = 1e-6

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
    
    def fit_closed_form(self, X, y):
        """Fit linear model via a stable least-squares solve.

        Solves for w in: minimize ||[X  1] w - y||^2,
        where the last element of w is the intercept term.
        (Equivalently, w = (X_aug^T X_aug)^{-1} X_aug^T y when the inverse exists.)

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Feature matrix.
        y : array_like of shape (N,) or (N, 1)
            Target values.

        Sets
        ----
        coef_ : ndarray of shape (D,)
            Learned weights (no intercept).
        intercept_ : float
            Learned intercept (bias).
        w : ndarray of shape (D + 1,)
            Concatenation of coef_ and intercept_.

        Raises
        ------
        ValueError
            If X and y have mismatched first dimensions.
        """

        X = self._ensure_2D(X)
        y = np.asarray(y, dtype=np.float64).ravel()  # ensure (N,)
        self.N, self.D = X.shape[0], X.shape[1]
         # Augment with ones for intercept
        X_aug = np.concatenate([X, np.ones([self.N, 1])], axis = 1)
        # Use lstsq (more stable than inv(X^T X) @ X^T y)
        w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.w = w
        # Split into coef and intercept
        self.coef_ = self.w[:-1]
        self.intercept_ = self.w[-1]
        

    def fit_gd(self, X, y, lr = 1e-2, max_iter = 1000, tol = 1e-8, l2 = 0.0, random_state = 0):
        """
            Full-batch gradient descent on MSE with optional L2 (Ridge) on weights.
            Minimizes: (1/N)||Xw + b - y||^2 + l2 * ||w||^2   (no penalty on b)
        """
         
        X = self._ensure_2D(X).astype(np.float64, copy=False)
        y = np.asarray(y, dtype=np.float64).ravel()
        N, D = X.shape
        self.N, self.D = N, D

         # 2) Standardize features (and keep transform for predict)
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std  = X.std(axis=0, keepdims=True) + 1e-12
        Xn = (X - self.x_mean) / self.x_std

        # 3) Center target so bias doesn’t need to fight large offsets
        self.y_mean = y.mean()
        yn = y - self.y_mean

        # Initialize small weights
        rng = np.random.default_rng(random_state)
        w = rng.normal(scale=1e-3, size=D)
        b = 0.0

        for it in range(max_iter):
            # Residuals
            y_hat = Xn @ w + b
            e = y_hat - yn
            
            # Gradients (MSE) + L2 on w only
            grad_w  =  2/ N * Xn.T @ e + 2.0 * l2 * w
            grad_b  = 2 / N * e.sum()

            # Update parameters
            step_w = lr * grad_w
            step_b = lr * grad_b
            w -= step_w
            b -= step_b

            # Early stopping on update size (L2 norm of concatenated step)
            if np.linalg.norm(np.r_[step_w, step_b]) < tol:
                break
        # 4) Map (w,b) back to original feature space and save
        coef = (w / self.x_std.ravel())               # shape (D,)
        intercept = float(b - (self.x_mean @ (w / self.x_std.ravel())).item() + self.y_mean)

        # 5) Save params
        self.coef_ = coef
        self.intercept_ = intercept
        self.w = np.r_[coef, intercept]
        return self

    def predict(self, X):
        """Predict targets for X.

        Parameters
        ----------
        X : array_like of shape (N, D) or (N,)
            Feature matrix or single feature column.

        Returns
        -------
        y_pred : ndarray of shape (N,)
            Predicted values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If feature dimensionality does not match training.
        """
        if self.w is None:
            raise RuntimeError("Model is not fitted yet. Call fit_closed_form or fit_gd first.")
        X = self._ensure_2D(X)
        X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X_aug @ self.w

    def mse(self, y, y_pred):
        """Compute Mean Squared Error (MSE).

        Parameters
        ----------
        y : array_like of shape (N,) or (N, 1)
            Ground-truth targets.
        y_pred : array_like of shape (N,) or (N, 1)
            Predicted targets.

        Returns
        -------
        float
            Mean of squared errors, i.e., mean((y - y_pred)^2).

        Raises
        ------
        ValueError
            If `y` and `y_pred` have different shapes after flattening.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if y.shape != y_pred.shape:
            raise ValueError("y and y_pred must have the same shape.")
        return float(np.mean((y.flatten() - y_pred.flatten())**2))