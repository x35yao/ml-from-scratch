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


def one_hot(y, n_classes, dtype=np.float32):
    y = np.asarray(y, dtype=np.int64).ravel()
    N = y.size
    Y = np.zeros((N, n_classes), dtype=dtype)
    Y[np.arange(N), y] = 1
    return Y

def one_hot_decode(Y):
    """
    Reverse of one_hot: convert one-hot matrix (N, K) back to labels (N,).
    """
    Y = np.asarray(Y)
    return np.argmax(Y, axis = 1)


def train_test_split(X, y=None, train_size=0.8, random_state=None, shuffle = True):
    rng = np.random.default_rng(random_state)
    N = len(X)
    perm = rng.permutation(N) if shuffle else np.arange(N)
    
    idx = int(N * train_size)
    train_idx, test_idx = perm[:idx], perm[idx:]
    
    if y is None:
        return X[train_idx], X[test_idx]
    else:
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def class_probs(y, n_classes):
    y = np.asarray(y, dtype=np.int64).ravel()
    counts = np.bincount(y, minlength=n_classes).astype(float)
    s = counts.sum()
    return counts / s if s > 0 else counts
 