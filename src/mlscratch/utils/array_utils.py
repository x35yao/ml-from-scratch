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

def add_intercept(X, position="last"):
    """
    Add an intercept (bias) column of 1s to X.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Input feature matrix.
    position : {"first", "last"}, default="first"
        If "first", add intercept as the first column.
        If "last", add intercept as the last column.

    Returns
    -------
    X_new : ndarray, shape (N, D+1)
        Feature matrix with intercept column.
    """
    X = ensure_2D(X)
    intercept = np.ones((X.shape[0], 1))

    if position == "first":
        return np.hstack([intercept, X])
    elif position == "last":
        return np.hstack([X, intercept])
    else:
        raise ValueError(f"position must be 'first' or 'last', got {position}")