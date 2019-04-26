import numpy as np #矩阵运算的库
import matplotlib.pyplot as plt #绘图库

x = np.linspace(0.0001, 0.9999, 3)
print(x)
P = np.log(x**1*(1-x)**3)
#plt.plot(x, P)
plt.scatter(x, P)
plt.show()

