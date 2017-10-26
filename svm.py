#!/usr/bin/python3

from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from tools.plot_tools import plot_decision_regions

#loading data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
#spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#standarizing data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#learning
clf = svm.SVC(kernel = 'linear', C = 1.0, random_state = 0)
clf.fit(X_train_std, y_train)

plot_decision_regions(X_train_std, y_train, clf) #test_idx = range(105, 150)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()