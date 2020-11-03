import numpy as np

a = [[11, 12, 5, 2], [15, 6, 10, 5], [10, 8, 12, 5], [12, 15, 8, 6]]
a = np.array(a)

b = np.array(np.linspace(1, 15, 15)).reshape(3, 5)
b = np.vstack((np.ones((1, 5)), b))

c = np.array(np.linspace(1, 4, 4)).reshape(4, 1)
# print(c)
# print(c[np.newaxis,:])


c = [0, 10, 2, 3, 4, 5, 6, 7, 8, 9, 1]
c[c == 10] = 0;
# print(c)

all_theta = []
all_theta.extend([1,2,3])
all_theta.extend([4,5,6])
# print(np.array(all_theta).reshape(2,3))


a=np.array([[2,1],[100,10],[3,2]])
b=np.array([[3,4,2],[1,2,1]])
print(a@b)