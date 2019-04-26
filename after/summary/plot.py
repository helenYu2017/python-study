import numpy as np 
import matplotlib.pyplot as plt 


def N(x, mu, std):
    return 1/(np.sqrt(2*np.pi)*std)*np.exp(-(x-mu)**2/(2*std**2))

x1 = np.random.normal(4, 1,[3000])
x2 = np.random.random(3000)
print(np.mean(x1), np.std(x1))
plt.hist(x1, bins=30, normed=True)
plt.hist(x2, bins=30, normed=True)
p = np.linspace(-2, 5, 1000)
plt.plot(p, N(p, np.mean(x1), np.std(x1)))
plt.plot(p, N(p, np.mean(x2), np.std(x2)))
plt.show()
print(N(0.9, np.mean(x1), np.std(x1)),
 N(0.9, np.mean(x2), np.std(x2)))