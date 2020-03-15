import numpy as np
import matplotlib.pyplot as mplot

data = np.loadtxt('ex1data1.txt', delimiter=',')
# get the 0th column
x= data[:,0]
XT = np.transpose(x);
print(XT)
y=data[:,1]
