import matplotlib.pyplot as plt
import numpy as np

from week3.sigmoid import sigmoid


def cost(theta, X, y):
    m = len(y)
    thetat_x = np.dot(X, theta)
    h_theta_of_xi = sigmoid(thetat_x)
    j_of_theta = (1 / m) * np.sum(-np.multiply(y, np.log(h_theta_of_xi)) - np.multiply((1 - y), np.log(1 - h_theta_of_xi)))
    return j_of_theta


# from dJ/dth = (1/m)(XT*X*theta - yT*X)
def diffOfCost(theta, X, y):
    m = len(y)
    return (1 / m) * (np.dot(np.dot(X.T, X), theta) - np.dot(y.T , X))


def gd(theta, X, y):
    alpha = 0.01
    num = 10
    for i in range (0, num):
        jTheta = cost(theta, X, y)
        theta = theta - alpha * diffOfCost(theta, X, y) 
        plt.plot(jTheta, theta)
    plt.show()
    return theta

 
data = np.loadtxt('ex2data1.txt', delimiter=',')
y = data[:, 2]
m = len(y)
X = np.hstack((np.ones((m, 1)), data[:, 0: 2]))
nOfFeat = len(X[0])
y = y.reshape(m, 1)
theta = np.zeros((nOfFeat, 1))
# print(cost(theta, X, y))
print('------------------------------------------------')
print(gd(theta, X, y))
print('------------------------------------------------')

