import numpy as np
from scipy.io import loadmat

# 5000 training example and 20*20 pixel (400 features)
from week4.utils import displayData

mat = loadmat('ex3data1.mat')

X= mat['X'] #5000
y=mat['y'] #400
y[y==10] = 0

#select random 100 training examples
random_X_indexes=np.random.choice(y.size, 100, replace=False)
random_X=X[random_X_indexes,:]
print(random_X)
displayData(random_X)

exit(0)
