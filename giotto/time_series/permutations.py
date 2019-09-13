# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed

import numpy as np


class OrdinalRepresentation(BaseEstimator, TransformerMixin):
    """Transformer performing an argsort of each row in each array in a collection.
    Based on ideas in `arXiv:1904.07403 <https://arxiv.org/abs/1904.07403>`_.

    Parameters
    ----------
    len_vector : int, optional, default: 8
        Used for performance optimization by exploiting numpy's
        vectorization capabilities.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    """

    def __init__(self, len_vector=8, n_jobs=None):
        self.len_vector = len_vector
        self.n_jobs = n_jobs

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.
        """
        pass

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data.

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
        """For each array in X, argsort each row in ascending order.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray of int, shape (n_samples, n_points, d)
            The transformed array.

        """

        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        slice_indices = list(range(0, n_samples, self.len_vector)) + [
            n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs)(
            delayed(np.argsort)(X[slice_indices[i]: slice_indices[i + 1]],
                                axis=2)
            for i in range(n_slices))
        X_transformed = np.concatenate(X_transformed)

        return X_transformed


class Entropy(BaseEstimator, TransformerMixin):
    """Transformer calculating the Shannon entropy of each array in a
    collection, in the following sense: in each array, the set of distinct rows
    is regarded as a set of possible states, and the probability of each state
    is the relative frequency of that state within the array.

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

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.
        """
        pass

    def _entropy(self, X):
        X_counts = np.unique(X, axis=0, return_counts=True)[1].reshape((-1, 1))
        X_normalized = X_counts / np.sum(X_counts, axis=0).reshape((-1, 1))
        return - np.sum(np.nan_to_num(X_normalized * np.log(X_normalized)),
                        axis=0).reshape((-1, 1))

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data.

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
        """Calculate the entropy of each array in X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray of int, shape (n_samples, n_points)
            Array of entropies (one per array in X).

        """

        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs)(
            delayed(self._entropy)(X[i]) for i in range(n_samples))
        X_transformed = np.concatenate(X_transformed)
        return X_transformed
