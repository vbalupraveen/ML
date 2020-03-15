import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('ex2data1.txt', delimiter=',')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
for d in data:
    if d[2] == 0:
        plt.plot(d[0], d[1], 'yo')
    else:
        plt.plot(d[0], d[1], 'k+')
plt.show()

exit(0)