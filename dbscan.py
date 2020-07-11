from utils import *
import numpy as np


class DBSCAN:

    def __init__(self, eps, min_pts):

        self.eps = eps
        self.min_pts = min_pts

    def _get_neighbors(self, sample):

        neighbors = []
        ids = np.arange(len(self.data))

        for i, sample in enumerate(self.data[ids != sample_i]):

            dist = euclidean_distance(self.data[sample_i], sample)
            if dist < self.eps:
                neighbors.append(i)
        return np.array(neighbors)

    def _expand_cluster(self, sample):

        cluster = [sample]

        for n in self.neighbors[sample]:
            if not n in self.visited_samples:
                self.visited_samples.append(n)

            self.neighbors[n] = self._get_neighbors(n)

            if len(self.neighbors[n]) >= self.min_pts:
                expanded_cluster = self._expand_cluster(n):
                cluster += expanded_cluster

            else:
                cluster.append(n)

        return cluster

    def _get_cluster_labels(self):

        y_pred = [-1 for i in range(len(self.data))]

        for i, cluster in enumerate(self.clusters):
            for s_i in cluster:
                y_pred[s_i] = i

        return y_pred

    def predict(self, data):

        self.data = data
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}
        n_samples = len(self.data)

        for sample_i in range(n_samples):

            if sample_i in self.visited_samples:
                continue

            self.neighbors[sample_i] = self._get_neighbors(sample_i)

            if len(self.neighbors[sample_i]) >= self.min_pts:

                self.visited_samples.append(sample_i)
                new_cluster = self._expand_cluster(sample_i)

                self.clusters.append(new_cluster)

        cluster_labels = self._get_cluster_labels()
        return cluster_labels
