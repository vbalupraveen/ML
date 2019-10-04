import numpy as np

from week3.sigmoid import sigmoid


def cost(theta, x, y):
    m = len(y)
    ones = np.ones((m, 1))
    xi = np.transpose(np.hstack((ones, x)))
    theta_t = np.transpose(theta)
    thetat_x = np.dot(theta_t, xi)
    h_theta_of_xi = sigmoid(thetat_x)
    j_of_theta = (1 / m) * np.sum(-y * np.log(h_theta_of_xi) - (1 - y) * np.log(1 - h_theta_of_xi))
    return j_of_theta


def derivativeOfTheta(theta, x, y):
    m = len(y)
    ones = np.ones((m, 1))
    xi = np.transpose(np.hstack((ones, x)))
    thetaT = np.transpose(theta)
    thetaTOfX = np.dot(thetaT, xi)
    h_theta_of_xi = sigmoid(thetaTOfX)
    d_theta = (1 / m) * (np.sum(h_theta_of_xi - y) * xi)
    return d_theta


def gd(theta, x, y, no_of_iterations):
    for i in range(0, no_of_iterations):
        theta = theta - derivativeOfTheta(theta, x, y)
    return theta


data = np.loadtxt('ex2data1.txt', delimiter=',')
x = data[:, 0: 2]
y = data[:, 2]
m = len(y)
nOfFeat = len(x[0])
y = y.reshape(m, 1)
theta = np.zeros((nOfFeat + 1, 1))
print(cost(theta, x, y))
print('------------------------------------------------')
print(gd(theta, x, y, 100))
print('------------------------------------------------')

