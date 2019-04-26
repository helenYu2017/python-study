import numpy as np 
import matplotlib.pyplot as plt 

mat = np.random.random([1000, 2])
#仿射变换
o=np.pi/4
A = np.array([
    [np.cos(o), np.sin(o)],
    [-np.sin(o), np.cos(o)]
])
A = np.array([
    [2, 1],
    [0, 0.5]
])
C = np.dot(mat, A)#+np.array([1, 1])
plt.scatter(mat[:, 0], mat[:, 1])
plt.scatter(C[:, 0], C[:, 1])
plt.show() 