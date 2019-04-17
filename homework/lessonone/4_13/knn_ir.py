from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
#plt.rcParams['font.sans-serif']="SimHei"#
# (0) load training data
iris = load_iris()
X = iris.data
y = iris.target

# (1) test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# (2) Model training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# (3) Predict & Estimate the score
#y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))