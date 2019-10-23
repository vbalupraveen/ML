import numpy as np
from scipy.optimize import fmin_bfgs

from week3.sigmoid import sigmoid


def cost(theta, X, y):
    m = len(y)
    thetat_x = np.dot(X, theta)
    h_theta_of_xi = sigmoid(thetat_x)
    j_of_theta = (1 / m) * np.sum(-np.multiply(y, np.log(h_theta_of_xi)) - np.multiply((1 - y), np.log(1 - h_theta_of_xi)))
    return j_of_theta


def cost_derivative(theta, X, y):
    m = len(y)
    thetat_x = np.dot(X, theta)
    g_minus_y = sigmoid(thetat_x) - y
    theta = (1 / m) * np.dot(X.T, g_minus_y)
    return theta


# from xi = (xi-ui)/si
def feature_scaling(X):
    # take median on the column
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm


def gradient(alpha, iterations, theta, X, y):
    for _ in range(0, iterations):
        theta = theta - (alpha / m) * cost_derivative(theta, X, y)
    return theta;


def fmin_gradient(theta, X, y):
    return fmin_bfgs(f=cost, x0=theta.flatten(), fprime=cost_derivative, args=(X,y.flatten()))


data = np.loadtxt('ex2data1.txt', delimiter=',')
y = data[:, 2]
m = len(y)
X = np.hstack((np.ones((m, 1)), data[:, 0: 2]))
nOfFeat = len(X[0])
y = y.reshape(m, 1)
theta = np.zeros((nOfFeat, 1))
print('--------------cost function--------------')
print(cost(theta, X, y))
print('--------------differentiation of Cost--------------')
print(cost_derivative(theta, X, y))
print('--------------feature normalized training set--------------')
print(feature_scaling(X))
print('--------------gradient descent--------------')
print(gradient(0.1, 400, theta, X, y))
print('--------------fmin gradient descent--------------')
print(fmin_gradient(theta, X, y))
exit(0)
