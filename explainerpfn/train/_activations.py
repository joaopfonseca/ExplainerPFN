"""
This script contains the implementation of all activation functions used in
DAGs for synthetic data generation.
"""

import numpy as np


def identity(x):
    return x


def logarithm(x):
    return np.log(np.abs(x) + 1e-10)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def absolute(x):
    return np.abs(x)


def sine(x):
    return np.sin(x)


def hyperbolic_tangent(x):
    return np.tanh(x)


def rank(x, scale=True):
    ranks = np.argsort(np.argsort(x))
    if scale:
        ranks = ranks / ranks.max()
    return ranks


def square(x):
    return np.square(x)


def power(x, exponent=3):
    return np.power(x, exponent)


def smooth_relu(x, beta=1):
    return np.log(1 + np.exp(beta * x)) / beta


def step(x):
    threshold = x.mean()
    return np.where(x > threshold, 1, 0)


def modulo(x, divisor=2):
    return np.mod(x, divisor)


ACTIVATIONS = [
    identity,
    logarithm,
    sigmoid,
    absolute,
    sine,
    hyperbolic_tangent,
    rank,
    square,
    power,
    smooth_relu,
    step,
    modulo,
]
