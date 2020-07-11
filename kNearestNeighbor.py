import numpy as np
from utils import *


class KNNClassifier:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.labels_map = {}
        for ind, val in enumerate(np.unique(self.y_train)):
            self.labels_map[ind] = val
            self.y_train[np.where(self.y_train == val)] = ind

    def _vote(self, neighbors):
        neighbors = neighbors.astype('int64')
        return np.bincount(neighbors).argmax()

    def predict(self, X_test):

        y_test = np.empty(X_test.shape[0])
        for ind, sample in enumerate(X_test):
            idx = np.argsort([euclidean_distance(sample, i)
                              for i in self.X_train])[:self.k]
            k_nearest_neighbor = self.y_train[idx]
            y_test[ind] = k_nearest_neighbor[self._vote(k_nearest_neighbor)]

        result = np.array([self.labels_map[i] for i in y_test])
        return result
