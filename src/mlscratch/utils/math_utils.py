import numpy as np

def sigmoid(z):
    # numerically stable sigmoid
    z = np.clip(z, -709, 709)    # exp overflow guard for float64
    return 1.0 / (1.0 + np.exp(-z))