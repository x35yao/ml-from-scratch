import numpy as np

def sigmoid(z):
    # numerically stable sigmoid
    z = np.clip(z, -709, 709)    # exp overflow guard for float64, cannot subtrac z_max because sigmoid is not shift invariant
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z, axis = -1):
    # numerically stable softmax
    z = np.asarray(z, dtype=np.float64)            # or float32 if you prefer
    z_max = np.max(z, axis = axis, keepdims = True)
    e = np.exp(z - z_max)
    return e/(np.sum(e, axis = axis, keepdims = True))

def log_softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    z_max = np.max(z, axis=axis, keepdims=True)  # stabilize
    logsumexp = np.log(np.sum(np.exp(z - z_max), axis=axis, keepdims=True)) + z_max
    return z - logsumexp