import numpy as np
from giotto.diagrams import PairwiseDistance
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform, pdist
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class SubSpaceExtraction(BaseEstimator, TransformerMixin):
    def __init__(self,
                 dist_percentage=0.05,
                 k_min=10,
                 k_max=100,
                 metric="euclidean",
                 n_jobs=-1):
        self.n_jobs = n_jobs
        self.dist_percentage = dist_percentage
        self.k_min = k_min
        self.k_max = k_max
        self.metric = metric

    def _select_subspace(self, space, label, matrix_distances, ind_x):
        target_vector_dist = matrix_distances[ind_x]
        max_dist = np.max(target_vector_dist) * self.dist_percentage

        indexes = target_vector_dist < max_dist
        if np.sum(indexes) > self.k_max:
            indexes = np.argsort(target_vector_dist)[:self.k_max]
        elif np.sum(indexes) < self.k_min:
            indexes = np.argsort(target_vector_dist)[:self.k_min]

        return space[indexes], label[indexes]

    def fit_transform_resample(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """ The transform method takes as input an array of dimension (n_sample, n_features) and for each sample
        it creates the neighourood point clouds."""
        def compute_all_distances(X, metric):
            if metric == "euclidean":
                return squareform(pdist(X, metric))
            else:
                return PairwiseDistance(metric=metric, n_jobs=self.n_jobs).fit_transform(X)

        distance_matrix = compute_all_distances(X, self.metric)

        Xy_list = Parallel(n_jobs=self.n_jobs)(delayed(self._select_subspace)(X, y, distance_matrix, i) for i in range(len(X)))

        max_n_points = np.max([x[0].shape[0] for x in Xy_list])

        X_new_dims = list(X.shape)
        X_new_dims.insert(1, max_n_points)

        X_new = np.empty(X_new_dims)
        y_new = np.full((X.shape[0], max_n_points), np.nan)

        for i, element in enumerate(Xy_list):
            X_new[i, :len(element[0])] = element[0]
            X_new[i, len(element[0]):] = element[0][-1]
            y_new[i, :len(element[1])] = element[1]

        return X_new, (y_new, X)

