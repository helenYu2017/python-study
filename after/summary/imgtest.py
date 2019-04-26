import numpy as np 
import matplotlib.pyplot as plt 

mat = plt.imread("img/jin.jpg")
print(mat/255)
mat = mat / 255
mat_new = mat[:, :, 0]/3+mat[:, :, 1]/3+mat[:, :, 2]/3

# SVD奇异值分解
M, A, N = np.linalg.svd(mat_new)
# 转换为对角矩阵
A = np.diag(A)
M1 = M[:, :100]
A1 = A[:100, :100]
N1 = N[:100, :]
mat = np.dot(np.dot(M1, A1), N1)

plt.matshow(mat, cmap=plt.get_cmap("gray"))
plt.show() 
print(mat.shape)