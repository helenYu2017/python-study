import  numpy as np
from sklearn import linear_model


def feature(x):
    return np.array([x%3,x%5,x%15])

def feature_label(y):
    """
    :param y: the element in train set
    :return:  the label of y, 0:normal ,1:say fizz_buzz,2:say buzz,3:say fizz
    """
    if y%3:
        return np.array([3])
    elif y%5:
        return np.array([2])
    elif y%15:
        return np.array([1])
    else:
        return np.array([0])

train_start=0
train_end=10
print(np.sqrt(36*36-18*18))
x_train=np.array([feature(i) for i in range(train_start,train_end)])
y_train=np.array([feature_label(i) for i in range(train_start,train_end)])

test_start=100
test_end=200
x_test=np.array([feature(i) for i in range(test_start,test_end)])
y_test=np.array([feature_label(i) for i in range(test_start,test_end)])

LR=linear_model.LogisticRegression()
LR.fit(x_train,y_train)
score=LR.score(x_test,y_test)
#print(score)


