import numpy as np 
import matplotlib.pyplot as plt 

def CONV(X, Y):
    return np.mean((X-np.mean(X))*(Y-np.mean(Y)))
def rho(X, Y):
    return CONV(X, Y)/np.std(X)/np.std(Y)

x1 = np.random.normal(0, 1, [1000])
x2 = np.random.normal(0, 0.6, [1000])
print(rho(x1, x2))
plt.scatter(x1, x2)
plt.show()