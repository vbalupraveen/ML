from scipy.io import loadmat
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, Lambda=0.0):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)
    # Setup some useful variables
    m = y.size
    m1 = X.shape[0]
    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    # ====================== YOUR CODE HERE ======================
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m1, 1)), a1))
    a2 = sigmoid(a1 @ Theta2.T)
    for j in range(num_labels):
        J = J + sum(-y[:, j] * np.log(a2[:, j]) - (1 - y[:, j]) * np.log(1 - a2[:, j]))
    cost = 1 / m * J
    reg_J = cost + Lambda / (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    return J, grad
