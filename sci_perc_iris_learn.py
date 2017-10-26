#!/usr/bin/python3

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np
from tools.plot_tools import plot_decision_regions
import matplotlib.pyplot as plt

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
perc = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
perc.fit(X_train_std, y_train)

y_pred = perc.predict(X_test_std)
print("Misclassified samples count: %d" % (y_pred != y_test).sum())

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=perc, test_idx=range(105,150))
plt.show()

