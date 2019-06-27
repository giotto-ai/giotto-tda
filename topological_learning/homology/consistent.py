import numpy as np
import sklearn as sk
from numpy.random.mtrand import RandomState

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils._joblib import Parallel, delayed
import math as m
import itertools


class ConsistentRescaling(BaseEstimator, TransformerMixin):
    def __init__(self, metric_kwargs={'metric': 'euclidean'}, n_neighbors=1, n_jobs=1):
        self.metric_kwargs = metric_kwargs
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'metric_kwargs': self.metric_kwargs, 'n_neighbors': self.n_neighbors, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    @staticmethod
    def _consistent_homology_distance(X, n_neighbors):
        indices_k_neighbors = np.argsort(X)[:, n_neighbors]
        distance_k_neighbor = X[np.arange(X.shape[0]), indices_k_neighbors]
        # Only calculate metric for upper triangle
        out = np.zeros(X.shape)
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            X_consistent[i, j] = X[i, j] / (m.sqrt(distance_k_neighbor[i]*distance_k_neighbor[j]))
        return X_consistent + X_consistent.T

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params()

        self.is_fitted = True
        return self

    #@jit
    def transform(self, X, y=None):
        """ Implementation of the sk-learn transform function that samples the input.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted'])

        metrics_kwargs = self.metrics_kwargs.copy()
        metric = metric_kwargs.pop('metric')

        X = pairwise_distances(X, metric=metric, n_jobs=self.n_jobs, **metric_kwargs)

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._consistent_homology_distance)(X[i, :, :], self.n_neighbors)
                                                              for i in range(X.shape[0]) )

        return X_transformed
