import numpy as np

def standardize(X):
    """
    Standardize features to zero mean and unit variance.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Input feature matrix.

    Returns
    -------
    Xs : ndarray, shape (N, D)
        Standardized features.
    mean : ndarray, shape (D,)
        Per-feature mean of original X.
    std : ndarray, shape (D,)
        Per-feature std of original X (with zeros replaced by 1.0).
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    # avoid division by zero → if std=0, feature is constant
    std[std == 0.0] = 1.0

    Xs = (X - mean) / std
    return Xs, mean, std
