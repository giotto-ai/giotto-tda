import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed

import numpy as np


class PermutationEmbedder(BaseEstimator, TransformerMixin):
    """

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

    def __init__(self, len_vector=8, n_jobs=1):
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

        n_samples = X.shape[0]

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(np.argsort) (X[slice_indices[i] : slice_indices[i+1]], axis=2)
                                                       for i in range(n_slices) )
        X_transformed = np.concatenate(X_transformed)

        return X_transformed


class PermutationEntropy(BaseEstimator, TransformerMixin):
    def __init__(self, len_vector=8, n_jobs=1):
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

        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._permutation_entropy) (X[i]) for i in range(n_samples) )
        X_transformed = np.concatenate(X_transformed)
        return X_transformed
