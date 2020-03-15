import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


# cut range into pieces
# X = np.linspace(-5, 5, 100)
# plt.plot(X, sigmoid(X))
# plt.xlabel('Random values from +ve to -ve')
# plt.ylabel('Sigmoid fun results')
# plt.text(2, 0.8, r'$\sigma(X)=\frac{1}{1+e^{-X}}$', fontsize=12)
# plt.show()
