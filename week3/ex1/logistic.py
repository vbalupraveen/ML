import numpy as np
from scipy.optimize import fmin_bfgs

from week3.ex1.sigmoid import sigmoid


def fmin_gradient(theta, X, y):
    return fmin_bfgs(f=cost, x0=theta.flatten(), fprime=gd, args=(X, y.flatten()))


def cost(theta, X, y):
    m = len(y)
    return (-1 / m) * (y @ np.log(sigmoid(X @ theta)) + (1 - y) @ np.log(1 - sigmoid(X @ theta)))

def gd(theta, X, y):
    m = len(y)
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))


# from xi = (xi-ui)/si
def feature_scaling(X):
    # take median on the column
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm


def optimize_gd(alpha, iterations, theta, X, y):
    for _ in range(0, iterations):
        theta = theta - (alpha) * gd(theta, X, y)
    return theta


def init_values(data):
    y = data[:, 2]
    m = len(y)
    X = data[:, 0: 2]
    X = np.hstack((np.ones((m, 1)), X))
    nOfFeat = len(X[0])
    theta = np.zeros((nOfFeat, 1))
    return theta, X, y


def predict(theta, X, y, scores):
    theta = fmin_gradient(theta, X, y)
    X_score = [1, scores[0], scores[1]]
    return sigmoid(np.dot(X_score, theta))
