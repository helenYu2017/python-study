import numpy as np 
import matplotlib.pyplot as plt 

def model(x, a, b):
    return a*x+b 

def gradf(x, d, a, b):
    y = model(x, a, b)
    grada = 2*(y-d)*x
    gradb = 2*(y-d)
    return [grada, gradb]

x = np.random.normal(1, 1, [1000])
d = x**2+2*x + np.random.normal(0, 0.4, [1000])

a = 1 
b = 1
batch_size = 32#批学习
eta = 0.3
for itr in range(500):
    idx = np.random.randint(0, 1000, batch_size)
    xin = x[idx]
    din = d[idx]
    ga, gb = gradf(xin, din, a, b)
    ga = np.mean(ga)
    gb = np.mean(gb)
    a = a - ga * eta 
    b = b - gb * eta 
print(a, b)

xplt = np.linspace(-1, 5, 1000)
ypred = model(xplt, a, b)

plt.scatter(x, d)
plt.plot(xplt, ypred, lw=3)
plt.show()