import numpy as np

def gini(p):
    p = np.asarray(p, dtype = float).ravel()
    return 1.0 - np.sum(p**2)

def entropy(p):
    p = np.asarray(p, dtype=float).ravel()
    m = p > 0  # mask to avoid log(0)
    return -np.sum(p[m] * np.log2(p[m]))  # use log2 for bits