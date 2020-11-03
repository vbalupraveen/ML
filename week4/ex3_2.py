from scipy.io import loadmat
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predition(X, theta1, theta2):
    a1 = sigmoid(X @ theta1)
    m1 = X.shape[0]
    a1 = np.hstack((np.ones((m1, 1)), a1))
    a2 = sigmoid(a1 @ theta2)
    # a2(5000x10) need to find index of max value in a row i.e the index has max probability (since index starts from 0 add 1)
    p = np.argmax(a2, axis=1) + 1
    return p


data = loadmat('ex3weights.mat')
mat = loadmat('ex3data1.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']
X = mat['X']
# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
# ‘0’ digit is labeled as ‘10’ in this dataset
y = mat['y']
y[y == 10] = 0

m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
pred = predition(X, Theta1.T, Theta2.T)
print("Training Set Accuracy:", sum(pred[:, np.newaxis] == y)[0] / 5000 * 100, "%")
