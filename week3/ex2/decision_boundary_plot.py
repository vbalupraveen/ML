import matplotlib.pyplot as plt
import numpy as np

from week3.ex2.reg_logistic import mapFeature, reg_fmin_gradient, loadData

data = np.loadtxt('ex2data2.txt', delimiter=',')
for d in data:
    if d[2] == 0:
        plt.scatter(d[0], d[1], c="r", marker="x", label="Rejected")
    else:
        plt.scatter(d[0], d[1], c="b", marker="+", label="Accepted")
X, y = loadData(data)
X = mapFeature(X, 6)
theta = np.zeros((len(X[0]), 1))
_lamba_ = 1
theta = reg_fmin_gradient(theta, X, y, _lamba_)


###################decision boundary###################
def map_feature_plot(x1, x2, degree):
    X = np.ones(1)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            terms = (x1 ** (i - j) * x2 ** j)
            X = np.hstack((X, terms))
    return X


m = 50
# take random (x1,x2) values
plt_x1 = np.linspace(-1, 1.5, m)
plt_x2 = np.linspace(-1, 1.5, m)
newX = np.stack((plt_x1, plt_x2), axis=1)
z = np.zeros((len(plt_x1), len(plt_x2)))
# calculate decision boundary for all points (x1,x2)
# decision boundary is X@theta
for i in range(len(plt_x1)):
    for j in range(len(plt_x2)):
        z[i][j] = map_feature_plot(plt_x1[i], plt_x2[j], 6) @ theta
# contour plot visualizes the relationship between three variables
plt.contour(plt_x1, plt_x2, z,0)

plt.show()
exit(0)
