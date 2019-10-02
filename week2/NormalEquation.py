import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', delimiter=',')
tempX = data[:, 0]
y = data[:, 1]

X = np.ones((len(tempX), 2))
for i in range(len(tempX)):
    X[i][1] = tempX[i]
XT = np.transpose(X)
XTX = np.matmul(XT, X)
inv = np.linalg.inv(XTX)
XTy = np.matmul(XT, y)
theta = np.matmul(inv, XTy)
print(theta)

x_line = np.linspace(10, 30,25)
y_line = theta[0] + theta[1] * x_line
print(y_line)

# visualize
plt.plot(tempX, y, 'kx')
plt.plot(x_line, y_line)
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()
