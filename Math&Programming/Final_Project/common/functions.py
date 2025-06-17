# coding: utf-8
import numpy as np

# Identity function (returns the input as-is)
def identity_function(x):
    return x

# Step function (binary thresholding)
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

# Derivative of the sigmoid function
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# ReLU (Rectified Linear Unit) activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU function
def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad

# Softmax function (with numerical stability for both 1D and 2D inputs)
def softmax(x):
    if x.ndim == 2:
        # For batch inputs
        x = x.T
        x = x - np.max(x, axis=0)  # Prevent overflow
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    # For single sample
    x = x - np.max(x)  # Prevent overflow
    return np.exp(x) / np.sum(np.exp(x))

# Mean Squared Error loss function
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # If train data is one-hot vector, return index of the answer lable
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    eps = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] +eps)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)