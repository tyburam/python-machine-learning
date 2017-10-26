#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuro.perceptron import Perceptron
import tools.plot_tools as ptools

iris_data = pd.read_csv('./data/iris.data', header = None)
y = iris_data.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = iris_data.iloc[0:100, [0,2]].values

perc = Perceptron(0.1, 10)
perc.fit(X,y)

#plt.plot(range(1, len(perc.errors_) + 1), perc.errors_, marker='o')
#plt.xlabel("Epochs")
#plt.ylabel("Number of misclasifications")

ptools.plot_decision_regions(X, y, perc)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()
