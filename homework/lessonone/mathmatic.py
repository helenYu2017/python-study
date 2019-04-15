import numpy as np
import matplotlib.pyplot as plot
import matplotlib  as matlib
matlib.rcParams['font.sans-serif']="SimHei"


def load_data(filename):
    data=np.load(filename)
    return data['X'],data['d']


def func(x,w):
    """
    :param x: input
    :param w: trainable params
    :return: the value of the function
    """
    a,b=w
    return a*x+b


def grad(x,d,w):
    """
    partial derivative of the trainable params
    :param x: input
    :param d: label
    :param w: trainable params
    :return:
    """
    a,b=w
    y=func(x,w)
    grad_a=2*(y-d)*x
    grad_b=2*(y-d)
    return grad_a,grad_b


def svd(X,D):
    w = [0, 0]
    eta = 0.1
    for itr in range(1000):
        index = np.random.randint(0, len(X))
        x = X[index]
        d = D[index]
        ga, gb = grad(x, d, w)
        w[0] -= eta * ga
        w[1] -= eta * gb
    return w


def pca(X, D):
    w = [0, 0]
    eta = 0.1
    batchsize = 10
    for itr in range(100):
        sum_ga,sum_gb=0,0
        for _ in range(batchsize):
            index = np.random.randint(0, len(X))
            x = X[index]
            d = D[index]
            ga, gb = grad(x, d, w)
            sum_ga += ga
            sum_gb += gb
        sum_ga = sum_ga/batchsize
        sum_gb = sum_gb/batchsize
        w[0] -= eta*sum_ga
        w[1] -= eta*sum_gb
    return w


def draw_scatter(x,y):
    plot.scatter(x[:, 0], y[:, 0], s=20, alpha=0.4, label="数据散点")


def draw_plot(x,y,color,label):
    plot.plot(x, y, lw=5, color=color, alpha=0.5, label=label)


if( __name__  == "__main__"):
    X,D=load_data('homework.npz')
    draw_scatter(X,D) # draw train data
    # the first method
    W = svd(X, D)     # train model using the SVD
    x1=np.linspace(-2,4,100) # generater the test data
    y1=func(x1,W)            # predict the result of the test data
    draw_plot(x1,y1,"#990000","svd预测关系")         # draw the
    #plot.legend()
   # plot.show()
    #the second

    W = pca(X, D)  # train model using the PCA
    x2 = np.linspace(-2, 4, 100)  # generater the test data
    y2 = func(x2, W)  # predict the result of the test data
    draw_plot(x2, y2, "009900","pca预测关系")  # draw the
    plot.legend()
    plot.show()




