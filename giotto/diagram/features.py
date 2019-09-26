# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD

import numpy as np

from sklearn.utils.validation import check_is_fitted
from ..utils.validation import check_diagram

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed


class PersistentEntropy(BaseEstimator, TransformerMixin):
    """Transformer for the calculation of persistent entropy from collections of
    persistence diagrams. Given a generic persistence diagram consisting of
    birth-death pairs (b, d), its persistent entropy is simply the (base e) entropy
    of the collection of differences d - b, normalized by the sum of all such differences.

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

    def _validate_params(self):
        """A class method that checks whether the hyperparameters and the input parameters
        of the :meth:`fit` are valid.
        """
        pass

    def _persistent_entropy(self, X):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_normalized = X_lifespan / np.sum(X_lifespan, axis=1).reshape((-1, 1))
        return - np.sum(np.nan_to_num(X_normalized * np.log(X_normalized)), axis=1).reshape((-1, 1))

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers
            d representing homology dimensions, and whose values are ndarrays of
            shape (n_samples, M_d, 2) whose each entries along axis 0 are persistence
            diagrams with M_d persistent topological features. For example, X
            could be the result of applying the ``transform`` method of a
            ``VietorisRipsPersistence`` transformer to a collection of point
            clouds/distance matrices, but only if that transformer was instantiated
            with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._validate_params()
        X = check_diagram(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each key in the dictionary X and for each persistence diagram in the
        corresponding ndarray, computes that diagram's persistent entropy. All results
        are arranged into an ndarray of appropriate shape.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers
            d representing homology dimensions, and whose values are ndarrays of
            shape (n_samples, M_d, 2) whose each entries along axis 0 are persistence
            diagrams with M_d persistent topological features. For example, X
            could be the result of applying the ``transform`` method of a
            ``VietorisRipsPersistence`` transformer to a collection of point
            clouds/distance matrices.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_X_keys)
            Array of persistent entropies (one value per sample and per key in X).

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        X = check_diagram(X)

        n_samples = X[next(iter(X.keys()))].shape[0]
        n_dimensions = len(X.keys())

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs)(delayed(self._persistent_entropy)(X[dimension][slice_indices[i]: slice_indices[i + 1]])
                                                     for dimension in X.keys() for i in range(n_slices))

        X_transformed = np.hstack([np.concatenate([X_transformed[i * n_slices + j] for j in range(n_slices)], axis=0) for i in range(n_dimensions)])
        return X_transformed
