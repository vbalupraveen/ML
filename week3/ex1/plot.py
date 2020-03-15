import matplotlib.pyplot as plt
import numpy as np

from week3.ex1.logistic import fmin_gradient, init_values

data = np.loadtxt('ex2data1.txt', delimiter=',')

theta, X, y = init_values(data)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

theta = fmin_gradient(theta, X, y)
#to plot graph we need to choose two point on the plane 2 (x1, x2)s where x1=score 1 and x2=score 2
# chose 2 random x1 vaues
plot_x1 = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
# since the prediction h_theta_x is always 0 or 1 and the decision boundary should between 0 and 1 so apply h_theta_x = 0.5 and find x2 values
# remember : z=theta_0+theta_1*x1+theta_2*x2
plot_x2 = -(theta[0] + theta[1] * plot_x1) / theta[2]
print(plot_x2)
for d in data:
    if d[2] == 0:
        plt.scatter(d[0], d[1], c="r", marker="x", label="Not Admitted")
    else:
        plt.scatter(d[0], d[1], c="b", marker="+", label="Admitted")

plt.plot(plot_x1, plot_x2, label="Decision_Boundary")
plt.show()

exit(0)