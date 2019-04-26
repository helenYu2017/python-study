from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X,y

def knn_test(X,y,random_state,k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # 评价模型
    accuracy = knn.score(X_test, y_test, )
    print('预测准确率:{:.2f}%'.format(accuracy * 100))

X,y=load_data()
for i in range(1,100):
    print(i)
    knn_test(X,y,i,i)



