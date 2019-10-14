"""Features from time series."""
# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted, check_array


class PermutationEntropy(BaseEstimator, TransformerMixin):
    """Transformer calculating the Shannon entropy of each array in a
    collection, in the following sense: in each array, the set of distinct rows
    is regarded as a set of possible states, and the probability of each state
    is the relative frequency of that state within the array.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _entropy(self, X):
        Xo = np.unique(X, axis=0, return_counts=True)[1].reshape((-1, 1))
        Xo = Xo / np.sum(Xo, axis=0).reshape((-1, 1))
        return -np.sum(np.nan_to_num(Xo * np.log(Xo)), axis=0).reshape((-1, 1))

    def _permutation_entropy(self, X):
        Xo = np.argsort(X, axis=2)
        Xo = np.stack([self._entropy(Xo[i]) for i in range(Xo.shape[0])])
        return Xo.reshape((-1, 1))

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
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

        """
        check_array(X, allow_nd=True)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Calculate the entropy of each array in `X`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray of int, shape (n_samples, n_points)
            Array of entropies (one per array in `X`).

        """

        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        X = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._permutation_entropy)(X[s])
            for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)
        return Xt
