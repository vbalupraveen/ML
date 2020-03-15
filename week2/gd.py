import numpy as np
import matplotlib.pyplot as plt

# population, profit
data = np.loadtxt('ex1data1.txt', delimiter=',')

x, y = data[:, 0], data[:, 1]

m = y.size  # number of examples
n = 2       # number of features

X = np.array([np.ones(m), x]).transpose()
y = np.reshape(y, (m, 1))
theta = np.zeros((n, 1))

iterations = 1500
alpha = 0.01

def computeCost(X, y, theta):
    m, J = y.size, 0
    return (1 / (2 * m)) * np.sum((np.dot(X, theta) - y)**2)

def gradientDescent(X, y, theta, alpha, num_iters):
    for i in range(0, num_iters):
        theta = theta - (alpha / m) * \
                        np.dot(X.transpose(), np.dot(X, theta) - y)
    return theta

J = computeCost(X, y, theta)
print(J)  # should be 32.07

theta = gradientDescent(X, y, theta, alpha, iterations)
x_line = np.linspace(5, 25)
y_line = theta[0] + theta[1] * x_line
              
# visualize
plt.plot(x, y, 'kx')
plt.plot(x_line, y_line)
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()