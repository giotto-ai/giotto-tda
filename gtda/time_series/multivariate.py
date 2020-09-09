"""Processing of multivariate time series."""
# License: GNU AGPLv3

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from ..utils import validate_params
from ..utils._docs import adapt_fit_transform_docs


@adapt_fit_transform_docs
class PearsonDissimilarity(BaseEstimator, TransformerMixin):
    """Pearson dissimilarities from collections of multivariate time series.

    The sample Pearson correlation coefficients between pairs of components of
    an :math:`N`-variate time series form an :math:`N \\times N` matrix
    :math:`R` with entries

    .. math:: R_{ij} = \\frac{ C_{ij} }{ \\sqrt{ C_{ii} C_{jj} } },

    where :math:`C` is the covariance matrix. Setting :math:`D_{ij} =
    (1 - R_{ij})/2` or :math:`D_{ij} = 1 - |R_{ij}|` we obtain a dissimilarity
    matrix with entries between 0 and 1.

    This transformer computes one dissimilarity matrix per multivariate time
    series in a collection. Examples of such collections are the outputs of
    :class:`SlidingWindow`.

    Parameters
    ----------
    absolute_value : bool, default: ``False``
        Whether absolute values of the Pearson correlation coefficients should
        be taken. Doing so makes pairs of strongly anti-correlated variables as
        similar as pairs of strongly correlated ones.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    See also
    --------
    SlidingWindow, gtda.homology.VietorisRipsPersistence

    """

    _hyperparameters = {'absolute_value': {'type': bool}}

    def __init__(self, absolute_value=False, n_jobs=None):
        self.absolute_value = absolute_value
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_observations, n_features)
            Input data. Each entry along axis 0 is a sample of ``n_features``
            different variables, of size ``n_observations``.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, allow_nd=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute Pearson dissimilarities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_observations, n_features)
            Input data. Each entry along axis 0 is a sample of ``n_features``
            different variables, of size ``n_observations``.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, n_features)
            Array of Pearson dissimilarities.

        """
        check_is_fitted(self, '_is_fitted')
        X = check_array(X, allow_nd=True)

        Xt = np.empty((X.shape[0], X.shape[2], X.shape[2]))
        for i, sample in enumerate(X):
            Xt[i, :, :] = np.corrcoef(sample, rowvar=False)
        Xt = 0.5 - Xt/2 if not self.absolute_value else 1 - np.abs(Xt)

        return Xt
