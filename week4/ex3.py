import numpy as np
from scipy.io import loadmat
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

mat = loadmat('ex3data1.mat')

X= mat['X']
y=mat['y']
y[y==10] = 0
print(X.shape)
print(y)

fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F"), cmap="hot") #reshape back to 20 pixel by 20 pixel
        axis[i,j].axis("off")
exit(0)

