# example code to be adopted for the ongoing project OCR
http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)       

clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(
   clf, iris.data, iris.target, cv=5)

scores   

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn import metrics
scores = cross_validation.cross_val_score(clf, iris.data, iris.target,
    cv=5, scoring='f1_weighted')
scores                                              


