# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import gen_even_slices
from ..utils.validation import check_diagram
from ._utils import _subdiagrams


class PersistentEntropy(BaseEstimator, TransformerMixin):
    """Transformer for the calculation of persistent entropy from collections
    of persistence diagrams. Given a generic persistence diagram consisting of
    birth-death-dimension tuples (b, d, k), its k-persistent entropy is simply
    the (base e) entropy of the collection of differences d - b for points of
    homology dimension k, normalized by the sum of all such differences.

    Parameters
    ----------
    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless in
        a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    """
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _persistent_entropy(self, X):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_normalized = X_lifespan / np.sum(X_lifespan, axis=1).reshape(-1, 1)
        return - np.sum(np.nan_to_num(
            X_normalized * np.log(X_normalized)), axis=1).reshape(-1, 1)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, 3)
            Input data. Array of persistence diagrams each of them containing
            a collection of points representing persistence feature through
            their birth, death and homology dimension.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        X = check_diagram(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each persistence subdiagram corresponding to an homology
        dimension k, computes that subdiagram's persistent entropy.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, 3)
            Input data. Array of persistence diagrams each of them containing
            a collection of points representing persistence feature through
            their birth, death and homology dimension.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_X_keys)
            Array of persistent entropies (one value per sample and per key in X).

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        X = check_diagram(X)

        homology_dimensions = sorted(list(set(X[0, :, 2])))
        n_dimensions = len(homology_dimensions)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._persistent_entropy)(_subdiagrams(X, [dim])[s, :, :2])
            for dim in homology_dimensions
            for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs)))
        n_slices = len(Xt) // n_dimensions
        Xt = np.hstack([np.concatenate([Xt[i * n_slices + j]
                                        for j in range(n_slices)], axis=0)
                        for i in range(n_dimensions)])
        return Xt
