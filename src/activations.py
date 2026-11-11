import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(x)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities