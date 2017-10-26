#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuro.adaline_gd import Adaline
import tools.plot_tools as ptools

iris_data = pd.read_csv('./data/iris.data', header = None)
y = iris_data.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = iris_data.iloc[0:100, [0,2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada1 = Adaline(0.001, 10)
ada1.fit(X, y)

ada2 = Adaline(0.0001, 10)
ada2.fit(X, y)

ada3 = Adaline(0.001, 10)
ada3.fit(X_std, y)

ada4 = Adaline(0.0001, 10)
ada4.fit(X_std, y)

fig, ax = plt.subplots(1, 4, figsize = (8, 4))

ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Cost")
ax[0].set_title('Adaline neuron eta = 0.001')

ax[1].plot(range(1, len(ada2.cost_) +1 ), np.log10(ada2.cost_), marker = 'o')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Cost")
ax[1].set_title('Adaline neuron eta = 0.0001')

ax[2].plot(range(1, len(ada3.cost_) +1 ), np.log10(ada3.cost_), marker = 'o')
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("Cost")
ax[2].set_title('Adaline neuron eta = 0.0001, std data')

ax[3].plot(range(1, len(ada4.cost_) +1 ), np.log10(ada4.cost_), marker = 'o')
ax[3].set_xlabel("Epochs")
ax[3].set_ylabel("Cost")
ax[3].set_title('Adaline neuron eta = 0.0001, std data')

#ptools.plot_decision_regions(X, y, adal)
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.legend(loc = 'upper left')
plt.show()
