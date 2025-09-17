import numpy as np

def ensure_2D(X):
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