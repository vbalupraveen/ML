import numpy as np

from week3.ex2.reg_logistic import loadData, mapFeature, reg_fmin_gradient, reg_cost

data = np.loadtxt('ex2data2.txt', delimiter=',')
X, y = loadData(data)
X = mapFeature(X, 6)
theta = np.zeros((len(X[0]), 1))
_lamba_ = 1
theta = reg_fmin_gradient(theta, X, y, _lamba_)
print(theta)

exit(0)
