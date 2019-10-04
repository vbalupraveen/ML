import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# cut range into pieces
# x = np.linspace(-5, 5, 100)
# plt.plot(x, sigmoid(x))
# plt.xlabel('Random values from +ve to -ve')
# plt.ylabel('Sigmoid fun results')
# plt.text(2, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=12)
# plt.show()
