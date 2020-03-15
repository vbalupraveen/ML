import numpy as np
import matplotlib.pyplot as plt

# population, profit
data = np.loadtxt('ex1data1.txt', delimiter=',')
x, y = data[:, 0], data[:, 1]
m = y.size
X = np.array([np.ones(m), x]).transpose()
theta = np.zeros((2, 1))

# print(x)
# print('--------------------')
# print(theta)
# print('--------------------')
# print(X)
# print('--------------------')
# print(np.dot(X, theta))

print((1 / (2 * m)) * np.sum((np.dot(X, theta) - y)**2))