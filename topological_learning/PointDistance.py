import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from umap.sparse import make_sparse_nn_descent
import umap.distance as dist
from sklearn.metrics.pairwise import paired_distances

from joblib import Parallel, delayed
import math as m
import itertools

import numpy as np



def consistent_homology_distance(X, metric, n_neighbors, **metric_kwargs):
    """Handle the callable case for pairwise_{distances,kernels}
    """
    X_sparse = csr_matrix(X)
    metric_nn_descent = make_sparse_nn_descent(metric, tuple(metric_kwargs.values()))
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    _, distances_kNeighbors = metric_nn_descent(X_sparse.indices, X_sparse.indptr, X_sparse.data, X.shape[0], n_neighbors, rng_state)
    distance_kNeighbor = distances_kNeighbors[:, -1]

    # Only calculate metric for upper triangle
    out = np.zeros((len(X), len(X)))
    iterator = itertools.combinations(range(len(X)), 2)
    for i, j in iterator:
        out[i, j] = paired_distances(X[i], X[j], metric=dist.normed_distance[metric], **metric_kwargs) / (m.sqrt(distances_kNeighbor[i]*distances_kNeighbor[j]))

    return out + out.T

def permutation_sequence_distance(x, y):
    pass


class PointDistance(BaseEstimator, TransformerMixin):
    implementedDistanceRecipes = {'consistent': consistent_homology_distance, 'permutation': permutation_sequence_distance}

    def __init__(self, distance_kwargs={'distance': 'consistent', 'metric': 'euclidian', 'n_neighbors': 3}, n_jobs=1):
        self.distance_kwargs = distance_kwargs
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'distance_kwargs': self.distance_kwargs, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def _pad(self, X):

    def fit(self, XList, y=None):
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

        self.distanceName = self.distance_kwargs['distance']
        self.distance = self.implementedDistanceRecipes[self.distanceName]

        self.isFitted = True
        return self

    #@jit
    def transform(self, XList, y=None):
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
        #check_is_fitted(self, ['isFitted'])

        X = XList[0]
        XListTransformed = []

        distance_kwargs = self.distance_kwargs.copy()
        if 'distance' in distance_kwargs:
            distance_kwargs.pop('distance')

        XTransformed = Parallel(n_jobs=self.n_jobs) ( delayed(self.distance)(X[i, :, :], **distance_kwargs)
                                                              for i in range(X.shape[0]) )


        if self.distanceName == 'permutation':
            XTransformed = self._pad(XTransformed)
        else:
            XTransformed = np.array(XTransformed)

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed
