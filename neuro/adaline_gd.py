import numpy as np
from .neuro_base import NeuroBase


class Adaline(NeuroBase):
    """Adaline neuron"""

    def fit(self, X, y):
        super().fit(X, y)
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label until unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

if __name__ == "__main__":
   print("Use with import")
