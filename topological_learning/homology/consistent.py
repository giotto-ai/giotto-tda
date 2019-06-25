import numpy as np
import sklearn as sk
from numpy.random.mtrand import RandomState

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from umap.sparse import make_sparse_nn_descent
import umap.distances as dist
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse import csr_matrix
from sklearn.utils._joblib import Parallel, delayed
import math as m
import itertools
from numba import jit




def consistent_homology_distance(X, metric, n_neighbors, **metric_kwargs):
    """Handle the callable case for pairwise_{distances,kernels}
    """
    distanceMatrix = sk.metrics.pairwise.euclidean_distances(X, X)
    indices_kNeighbors = np.argsort(distanceMatrix)[:, :n_neighbors]
    distances_kNeighbors = distanceMatrix[np.arange(distanceMatrix.shape[0])[:, None], indices_kNeighbors].copy()
    distance_kNeighbor = distances_kNeighbors[:, -1]

    # Only calculate metric for upper triangle
    out = np.zeros((len(X), len(X)))
    iterator = itertools.combinations(range(len(X)), 2)
    for i, j in iterator:
        out[i, j] = distanceMatrix[i, j] / (m.sqrt(distance_kNeighbor[i]*distance_kNeighbor[j]))

    return out + out.T

def permutation_sequence_distance(x, y):
    pass


class PointDistance(BaseEstimator, TransformerMixin):
    implementedDistanceRecipes = {'consistent': consistent_homology_distance, 'permutation': permutation_sequence_distance}

    def __init__(self, distance_kwargs={'distance': 'consistent', 'metric': 'euclidean', 'n_neighbors': 3}, n_jobs=1):
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
        pass

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
