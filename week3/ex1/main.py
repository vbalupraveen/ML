import numpy as np

from week3.ex1.logistic import init_values, predict, cost, gd, feature_scaling, optimize_gd, fmin_gradient

data = np.loadtxt('ex2data1.txt', delimiter=',')
theta, X, y = init_values(data)
# print('--------------cost function--------------')
# print(cost(theta, X, y))
# print('--------------differentiation of Cost--------------')
# print(gd(theta, X, y))
# print('--------------feature normalized training set--------------')
# print(feature_scaling(X))
# print('--------------gradient descent--------------')
# print(gradient(0.1, 1, theta, X, y))
print('--------------fmin gradient descent--------------')
print(fmin_gradient(theta, X, y))
# print('--------------predict--------------')
# print(predict(theta, X, y, np.array([45, 85])))

exit(0)
