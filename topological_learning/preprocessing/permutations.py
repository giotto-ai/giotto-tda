import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed

import numpy as np


class PermutationEmbedder(BaseEstimator, TransformerMixin):
    """Transformer returning a representation of a collection (typically, a time series)
    of point clouds in :math:`\\mathbb{R}^d` -- where each point cloud is an array
    of size (n_points, d) -- as a collection of arrays of the same shape, whose
    each row is the result of applying ``np.argsort`` to the corresponding row
    in the point cloud array. Based on ideas in `arXiv:1904.07403 <https://arxiv.org/abs/1904.07403>`_.

    Parameters
    ----------
    len_vector : int, optional, default: 8
        Used for performance optimization by exploiting numpy's vectorization capabilities.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless in
        a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    """

    def __init__(self, len_vector=8, n_jobs=None):
        self.len_vector = len_vector
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default: True
            Behaviour not yet implemented.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return {'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """

        self._validate_params()

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each point cloud in X, argsort each row, in ascending order.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray of int, shape (n_samples, n_points, d)
            The transformed array.

        """

        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(np.argsort) (X[slice_indices[i] : slice_indices[i+1]], axis=2)
                                                       for i in range(n_slices) )
        X_transformed = np.concatenate(X_transformed)

        return X_transformed


class PermutationEntropy(BaseEstimator, TransformerMixin):
    def __init__(self, len_vector=8, n_jobs=None):
        self.len_vector = len_vector
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
            of the :meth:'fit' are valid.
            """
        pass

    def _permutation_entropy(self, X):
        X_counts = np.unique(X, axis=0, return_counts=True)[1].reshape((-1, 1))
        X_normalized = X_counts / np.sum(X_counts, axis=0).reshape((-1, 1))
        return - np.sum(np.nan_to_num(X_normalized * np.log(X_normalized)), axis=0).reshape((-1, 1))

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

        self._is_fitted = True
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
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._permutation_entropy) (X[i]) for i in range(n_samples) )
        X_transformed = np.concatenate(X_transformed)
        return X_transformed
