import math as m
import numpy as np
import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
import itertools

from ._metrics import _parallel_pairwise, implemented_metric_recipes
from ._utils import _sample, _pad


class DiagramDistance(BaseEstimator, TransformerMixin):
    """
    Transformer for calculating distances between persistence diagrams.
    """
    def __init__(self, metric='bottleneck', metric_params={'order': np.inf}, n_jobs=1):
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'metric': self.metric, 'metric_params': self.metric_params, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        pass

    def fit(self, X, y=None):
        self._validate_params()

        if 'n_samples' in self.metric_params:
            self.n_samples = self.metric_params['n_samples']

        self._X = X

        if self.metric in ['landscape', 'betti']:
            self.metric_params['sampling'] = _sample(self._X, self.n_samples)

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['is_fitted'])
        n_diagrams_X = next(iter(X.values())).shape[0]

        metric_params = self.metric_params.copy()

        if 'n_samples' in metric_params:
            metric_params.pop('n_samples')

        is_same = np.sum([ np.array_equal(X[dimension], self._X[dimension]) for dimension in X.keys()]) == len(X)
        if is_same:
            # Only calculate metric for upper triangle
            iterator = list(itertools.combinations(range(n_diagrams_X), 2))
            X_transformed = _parallel_pairwise(X, X, self.metric, metric_params, iterator, self.n_jobs)
            X_transformed = X_transformed + X_transformed.T
        else:
            max_betti_numbers = { dimension: max(self._X[dimension].shape[1], X[dimension].shape[1]) for dimension in self._X.keys()}
            self._X = _pad(self._X, max_betti_numbers)
            X = _pad(X, max_betti_numbers)
            Y = { dimension: np.vstack([self._X[dimension], X[dimension]]) for dimension in self._X.keys() }
            n_diagrams_Y = next(iter(Y.values())).shape[0]

            # Calculate all cells
            iterator = tuple(itertools.product(range(n_diagrams_Y), range(n_diagrams_X)))
            X_transformed = _parallel_pairwise(Y, X, self.metric, metric_params, iterator, self.n_jobs)

        return X_transformed
