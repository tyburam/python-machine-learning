import numpy as np
from .neuro_base import NeuroBase


class Adaline(NeuroBase):
    """Adaline neuron - stohastic gradient"""

    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        super().__init__(eta, n_iter)
        self.w_initialized = False
        self.shuffle = shuffle

        if(random_state):
            np.random.seed(random_state)

    def fit(self, X, y):
        super().fit(X, y)
        self.cost_ = []
        self._initialize_weights(X.shape[1])

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def _shuffle(self, X, y):
        """Shuffe training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

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
