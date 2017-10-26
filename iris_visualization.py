#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris_data = pd.read_csv('./iris.data', header = None)
y = iris_data.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = iris_data.iloc[0:100, [0,2]].values

plt.scatter(X[:50,0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
