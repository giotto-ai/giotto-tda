import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from umap.sparse import make_sparse_nn_descent
from joblib import Parallel, delayed
import math as m
import itertools

import numpy as np

class ConsistentPointDistance(BaseEstimator, TransformerMixin):
    """
    Transformer for consistent persistent homology.

    Parameters
    ----------
    samplingType : str
        The type of sampling

        - data_type: string, must equal either 'points' or 'distance_matrix'.
        - data_iter: an iterator. If data_iter is 'points' then each object in the iterator
          should be a numpy array of dimension (number of points, number of coordinates),
          or equivalent nested list structure. If data_iter is 'distance_matrix' then each
          object in the iterator should be a full (symmetric) square matrix (numpy array)
          of shape (number of points, number of points), __or a sparse distance matrix

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, n_neighbors=3, metric='euclidean', n_jobs=1, **metric_kwargs):
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.n_jobs = n_jobs

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def consistent_metric(self, x, dist_x_kNeighbor, y, dist_y_kNeighbor):
        return self.metric(x, y, **self.metric_kwargs) / (m.sqrt(dist_x_kNeighbor*dist_y_kNeighbor))

    def _pairwise_callable(self, X):
        """Handle the callable case for pairwise_{distances,kernels}
        """
        #    X, Y = check_pairwise_arrays(X, Y)
            X_sparse = csr_matrix(X)
            _, distances_kNeighbors = metric_nn_descent(X_sparse.indices, X_sparse.indptr, X_sparse.data, X.shape[0], self.n_neighbors, rng_state)
            distance_kNeighbor = distances_kNeighbors[:, -1]

            # Only calculate metric for upper triangle
            out = np.zeros((len(X), len(X)), dtype='float')
            iterator = itertools.combinations(range(len(X)), 2)
            for i, j in iterator:
                out[i, j] = self.consistent_metric(X[i], distance_kNeighbor[i], X[j], distance_kNeighbor[j])

        return out + out.T

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

        numberOuterWindows = X.shape[0]
        numberInnerWindows = X.shape[1]
        metric_nn_descent = make_sparse_nn_descent(self.metric, self.metric_kwargs)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        XTransformed = np.empty((numberOuterWindows, numberInnerWindows, numberInnerWindows)

        XTransformed[i, :, :] = Parallel(n_jobs = self.n_jobs) (delayed(self._pairwise_callable)(X[i, :, :])
                                                                for i in range(numberOuterWindows))

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed
