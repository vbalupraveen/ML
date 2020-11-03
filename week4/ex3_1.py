import numpy as np
from scipy.io import loadmat
from week4.utils import displayData
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lrCostFunction(X, y, theta, Lambda):
    m = len(y);
    h_theta = sigmoid(X @ theta)
    error = (-y * np.log(h_theta)) - ((1 - y) * np.log(1 - h_theta))
    # cost function
    cost = 1 / m * np.sum(error)
    # regularized cost function
    # theta[1:]=all rows from 1st index
    regCost = cost + Lambda / (2 * m) * sum(theta[1:] ** 2)
    return regCost

# dJ/dtheta
def vectorizedGd(X, y, theta, Lambda):
    m = len(y)
    h_theta = sigmoid(X @ theta)
    theta_j = 1 / m * (X.transpose() @ (h_theta - y))[1:] + (Lambda / m) * theta[1:]
    theta_0 = 1 / m * (X.transpose() @ (h_theta - y))[0]
    grad = np.vstack((theta_0, theta_j))
    return grad


def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    return theta and the list of the cost of theta during each iteration
    """
    m = X.shape[0]
    J_history = []
    for i in range(num_iters):
        cost = lrCostFunction(X, y, theta, Lambda)
        grad = vectorizedGd(X, y, theta, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history


def oneVsAll(X, y, num_labels, Lambda):
    """
    Takes in numpy array of X,y, int num_labels and float lambda to train multiple logistic regression classifiers
    depending on the number of num_labels using gradient descent.
    Returns a matrix of theta, where the i-th row corresponds to the classifier for label i
    """
    m, n = X.shape[0], X.shape[1]
    # number of features + one for X0
    initial_theta = np.zeros((n + 1, 1))
    all_theta = []
    all_J = []
    # add intercept terms
    X = np.hstack((np.ones((m, 1)), X))

    for i in range(1, num_labels + 1):
        theta, J_history = gradientDescent(X, np.where(y == i, 1, 0), initial_theta, 1, 300, Lambda)
        all_theta.extend(theta)
        all_J.extend(J_history)
    return np.array(all_theta).reshape(num_labels, n + 1), all_J


def predictOneVsAll(all_theta, X):
    # all_theta (10x401) X (5000x401)
    m = X.shape[0];
    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))
    # predictions=probability of 400 pixel image turns in to a number
    predictions = sigmoid(X @ all_theta.T)
    print(predictions)
    # returns matrix of (5000x10) like 0.1,0.23, 0.49,.....,0.95 (10 values)
    # i.e the 0.95 =1 probability of image tends to 9th index (10 number) is high
    # np.argmax(array, 1) returns index of largest value in the row
    p = np.argmax(predictions, axis=1) + 1
    return p


# 5000 training example and 20*20 pixel (400 features)
mat = loadmat('ex3data1.mat')
X = mat['X']  # 5000
y = mat['y']  # 400
# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
# ‘0’ digit is labeled as ‘10’ in this dataset
num_labels = 10
y[y == 10] = 0

# select random 100 training examples
random_X_indexes = np.random.choice(y.size, 100, replace=False)
random_X = X[random_X_indexes, :]
displayData(random_X)
lambda_ = 0.1
# all_theta = theta for each label
all_theta, all_J = oneVsAll(X, y, num_labels, lambda_)

plt.plot(all_J[0:300])
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

pred = predictOneVsAll(all_theta, X)
print("Training Set Accuracy:", sum(pred[:, np.newaxis] == y)[0] / 5000 * 100, "%")

# -----------------------------------------------------------------------------
lambda_ = 3
theta_t = np.array([-2, -1, 1, 2]).reshape(4, 1)
X_t = np.array([np.linspace(0.1, 1.5, 15)]).reshape(3, 5).T
X_t = np.hstack((np.ones((5, 1)), X_t))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
J = lrCostFunction(X_t, y_t, theta_t, lambda_)
grad = vectorizedGd(X_t, y_t, theta_t, lambda_)
print("Cost:", J, "Expected cost: 2.534819")
print("Gradients:\n", grad, "\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")

exit(0)
