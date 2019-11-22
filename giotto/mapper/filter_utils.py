from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import pdist, squareform
import numpy as np


class Eccentricity(BaseEstimator, TransformerMixin):
    def __init__(self, exponent=np.inf, metric='euclidean', metric_params=None):
        self.exponent = exponent
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else dict()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        distance_matrix = squareform(
            pdist(X, metric=self.metric, **self.metric_params)
        )
        Xt = np.linalg.norm(distance_matrix, axis=1, ord=self.exponent)
        return Xt
