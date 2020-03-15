import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(x, y)
F = X ** 2 + Y ** 2 - 1.0
fig, ax = plt.subplots()
ax.contour(X, Y, F, [0])
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(linestyle='-')
plt.show()
exit(0)
