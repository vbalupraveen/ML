import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('ex2data2.txt', delimiter=',')
for d in data:
    if d[2] == 0:
        x = plt.scatter(d[0], d[1], c='r', marker='x', label='Rejected')
    else:
        o = plt.scatter(d[0], d[1], c='b', marker='o', label='Accepted')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend((x, o), ('Rejected', 'Accepted'))

plt.show()
exit(0)
