from pylab import *
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target
#cross_val_predict返回和`y`相同尺寸的数组
#每一个entry是通过交叉验证的相应预测
predicted = cross_val_predict(lr,boston.data,y,cv=10)

plt.rcParams['font.sans-serif']="SimHei"
#设置中文字体
myfont = None#matplotlib.font_manager.FontProperties(fname="Microsoft-Yahei-UI-Light.ttc")
mpl.rcParams['axes.unicode_minus'] = False
#绘制
plt.scatter(y,predicted)
plt.plot([y.min(),y.max()],[y.min(),y.max()],"k--",lw=4)
plt.title(u'绘制交叉验证预测')
plt.xlabel(u'测度')
plt.ylabel(u'预测')
#显示绘制结果
plt.show()