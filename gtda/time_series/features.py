"""Features from time series."""
# License: GNU AGPLv3

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted, check_array

from ..utils._docs import adapt_fit_transform_docs


@adapt_fit_transform_docs
class PermutationEntropy(BaseEstimator, TransformerMixin):
    """Entropies from sets of permutations arg-sorting rows in arrays.

    Given a two-dimensional array `A`, another array `A'` of the same size is
    computed by arg-sorting each row in `A`. The permutation entropy [1]_ of
    `A` is the Shannon entropy of the probability distribution given by
    the relative frequencies of each arg-sorting permutation in `A'`.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    See also
    --------
    TakensEmbedding, gtda.diagrams.PersistenceEntropy

    References
    ----------
    .. [1] C. Bandt and B. Pompe, "Permutation Entropy: A Natural Complexity
           Measure for Time Series"; *Phys. Rev. Lett.*, **88**.17, 2002;
           `doi: 10.1103/physrevlett.88.174102
           <https://doi.org/10.1103/physrevlett.88.174102>`_.

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _entropy(self, X):
        Xo = np.unique(X, axis=0, return_counts=True)[1].reshape(-1, 1)
        Xo = Xo / np.sum(Xo, axis=0).reshape(-1, 1)
        return -np.sum(np.nan_to_num(Xo * np.log2(Xo)), axis=0).reshape(-1, 1)

    def _permutation_entropy(self, X):
        Xo = np.argsort(X, axis=2)
        Xo = np.stack([self._entropy(x) for x in Xo])
        return Xo.reshape(-1, 1)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, allow_nd=True)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Calculate the permutation entropy of each two-dimensional array in
        `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of int, shape (n_samples, 1)
            One permutation entropy per entry in `X` along axis 0.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._permutation_entropy)(Xt[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)
        return Xt
