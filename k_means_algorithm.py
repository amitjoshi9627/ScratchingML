import numpy as np
from utils import *


class K_means:
    def __init__(self, k=3, max_iter=20, tolerance=0.001):

        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance

    def _init_random_centroids(self, data):

        self.centroids = []
        for i in range(self.k):
            self.centroids[i] = data[np.random.choice(range(len(data)))]

    def _closest_centroid(self, sample):

        closest_i = 0
        closest_dist = float('inf')
        for i, cent in enumerate(self.centroids):
            dist = euclidean_distance(cent, sample)
            if distance < closest_dist:
                closest_dist = distance
                closest_i = i
        return closest_i

    def _create_cluster(self, data):

        self.clusters = [[] for _ in range(self.k)]

        for i, sample in enumerate(data):

            centroid_i = self._closest_centroid(sample)
            self.clusters[centroid_i].append(i)

    def _calculate_centroids(self, data):

        for i, cluster in enumerate(self.clusters):

            centroid_i = np.mean(data[cluster], axis=0)
            self.centroids[i] = centroid_i

    def _get_cluster_labels(self, data):

        y_pred = [0 for _ in range(len(data))]

        for i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                y_pred[sample_i] = i
        return y_pred

    def predict(self, data):

        self._init_random_centroids(data)

        for _ in range(self.max_iterations):

            self._create_cluster(data)
            prev_centroids = self.centroids
            self._calculate_centroids(data)
            new_centroids = self.centroids

            diff = new_centroids - prev_centroids

            if diff < self.tolerance:
                break
        return self._get_cluster_labels(data)

npy_path = "/home/user/Documents/ML/Algorithms/tile_0_0_finalLabels_VH4_temp.npy"
data = np.load(npy_path)
print(data.shape)