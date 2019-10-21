import numpy as np

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


data = np.loadtxt('ex2data1.txt', delimiter=',')
y = data[:, 2]
m = len(y)
X = np.hstack((np.ones((m, 1)), data[:, 0: 2]))
nOfFeat = len(X[0])
y = y.reshape(m, 1)
theta = np.zeros((nOfFeat, 1))
print('------------------------------------------------')
print(cost(theta, X, y))
print(cost_derivative(theta, X, y))
print('------------------------------------------------')
exit(0)
