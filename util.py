import numpy as np


def unipolar_sigmoid(x):
    # σ(x) = 1 / (1 + e^(-x)).
    return 1 / (1 + np.exp(-x))


def unipolar_sigmoid_derivative(x):
    # σ(x)' = σ(x) * (1 - σ(x))
    sig_x = unipolar_sigmoid(x)
    return sig_x * (1 - sig_x)
