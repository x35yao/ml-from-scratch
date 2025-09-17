import numpy as np
from mlscratch.utils import ensure_2D, standardize, one_hot, log_softmax, add_intercept, one_hot_decode, softmax

class SoftmaxRegression:
    def __init__(self):
        self.w = None
        self.b = None
        self.classes_ = None
        self.x_mean = None
        self.x_std = None
        self.coef_ = None
        self.intercept_ = None
        self.W = None
    
    def fit(self, X, y, lr = 1e-2, max_iter = 1000, tol = 1e-8, random_state = 0):
        X = ensure_2D(X)
        y = np.asarray(y)
        if y.ndim != 1:
            y = y.ravel()

        # Ensure class labels are 0..K-1 (store original mapping)
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        K = len(classes)                         # number of classes
        N, D = X.shape

        # One hot
        yn = one_hot(y_idx, n_classes = K)                              # (N, K)
        
        Xn, self.x_mean, self.x_std = standardize(X)                # (N, D),  (D,), (D,)

        # Initialize 
        rng = np.random.default_rng(random_state)
        w = rng.normal(loc=0.0, scale=1e-2, size=(D, K))         # (D, K)        
        b = np.zeros(K)                                          # (K, )

        for it in range(max_iter):
            z = Xn @ w + b                                          # (N, K)
            logp = log_softmax(z, axis = 1)                         # (N, K)
            p = np.exp(logp)                                        # Back to probability

            grad_w =  Xn.T @ (p - yn) / N
            grad_b = np.mean(p - yn, axis = 0)

            w -= lr * grad_w
            b -= lr * grad_b

            # Convergence check
            gnorm = np.linalg.norm(np.concatenate([grad_w.ravel(), grad_b.ravel()]))
            if gnorm < tol:
                break
        
        # Map (w,b) back to original feature space and save
        coef = w / self.x_std[:, None]                      # (D, K)  
        intercept = b - (self.x_mean / self.x_std) @ w      # (K,)     

        # Save params
        self.coef_ = coef
        self.intercept_ = intercept
        # augmented weights matrix (for convenience): last row = intercept
        self.W = np.vstack([coef, intercept[None, :]])    # (D+1, K)
        return self   


    def predict_proba(self, X):
        X = ensure_2D(X).astype(np.float64, copy=False)
        if self.W is None:
            raise ValueError("Model not trained!")
        X_aug = add_intercept(X)                     # (N, D+1)
        z = X_aug @ self.W                          # (N, K)
        return softmax(z, axis=1)                    # (N, K)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]                    # map back to original labels

    def score(self, X, y):
        y = np.asarray(y).ravel()
        return np.mean(y == self.predict(X))




