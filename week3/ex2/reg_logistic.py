import numpy as np
from scipy.optimize import fmin_bfgs


def loadData(data):
    X = data[:, 0:2];
    y = data[:, 2]
    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# need higher order polynomial to predict decision boundary. Since we don't have enough date create data from X
# produces 1
# [ [x1], [x2],[x1^2],[x1x2],[x2^2],[x1^3], ...,[x1x2^5],x2^6]]
def mapFeature(X, degree):
    x1 = X[:, 0]
    x2 = X[:, 1]
    X = np.ones(len(x1)).reshape(len(x1), 1)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            terms = (x1 ** (i - j) * x2 ** j).reshape(len(x1), 1)
            X = np.hstack((X, terms))
    return X


def reg_cost(theta, X, y, _lambda_):
    m = len(y)
    J = (-1 / m) * (y @ np.log(sigmoid(X @ theta)) + (1 - y) @ np.log(1 - sigmoid(X @ theta)))
    # all rows except theta 0
    theta_square = theta[1:].T @ theta[1:]
    reg = (_lambda_ / (2 * m)) * (theta_square)
    J = J + reg
    return J


def reg_gd(theta, X, y, _lambda_):
    m = len(y)
    reg = (_lambda_ / m) * theta
    gd = 1 / m * (X.transpose() @ (sigmoid(X @ theta) - y)) + reg
    #should not penalize theta 0
    gd[0] = gd[0]-reg[0]
    return gd


def reg_fmin_gradient(theta, X, y, _lambda_):
    return fmin_bfgs(f=reg_cost, x0=theta.flatten(), fprime=reg_gd, args=(X, y.flatten(), _lambda_))
