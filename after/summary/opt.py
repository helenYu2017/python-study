
def f(x1, x2):
    return x1**2+x2**2+2*x1
def df(x1, x2):
    return [2*x1+2, 2*x2]
x1 = 3
x2 = 3
eta = 10#学习率
for itr in range(20):
    g1, g2 = df(x1, x2)
    x1 = x1 - g1 * eta
    x2 = x2 - g2 * eta
    print(x1, x2, f(x1, x2))