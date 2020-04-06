"""Filter functions commonly used with Mapper."""
# License: GNU AGPLv3

import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import entr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ..utils._docs import adapt_fit_transform_docs


@adapt_fit_transform_docs
class Eccentricity(BaseEstimator, TransformerMixin):
    """Eccentricities of points in a point cloud or abstract metric space.

    Let `D` be a square matrix representing distances between points in a
    point cloud, or directly defining an abstract metric (or metric-like)
    space. The eccentricity of point `i` in the point cloud or abstract
    metric space is the `p`-norm (for some `p`) of row `i` in `D`.

    Parameters
    ----------
    exponent : int or float, optional, default: ``2``
        `p`-norm exponent used to calculate eccentricities from the distance
        matrix.

    metric : str or function, optional, default: ``'euclidean'``
        Metric to use to compute the distance matrix if point cloud data is
        passed as input, or ``'precomputed'`` to specify that the input is
        already a distance matrix. If not ``'precomputed'``, it may be
        anything allowed by :func:`scipy.spatial.distance.pdist`.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function.

    """

    def __init__(self, exponent=2, metric='euclidean', metric_params=None):
        self.exponent = exponent
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method exists to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, \
            n_samples)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        # TODO: Consider making this transformer stateful so that the
        #  eccentricities of new points relative to the data seen in fit
        #  may be computed. May be useful for supervised tasks with Mapper?
        #  Evaluate performance impact of doing this.
        check_array(X)

        if self.metric_params is None:
            self.effective_metric_params_ = dict()
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        return self

    def transform(self, X, y=None):
        """Compute the eccentricities of points (i.e. rows) in  `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, \
            n_samples)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, 1)
            Column vector of eccentricities of points in `X`.

        """
        check_is_fitted(self)
        Xt = check_array(X)

        if self.metric != 'precomputed':
            Xt = squareform(
                pdist(Xt, metric=self.metric, **self.effective_metric_params_))

        Xt = np.linalg.norm(Xt, axis=1, ord=self.exponent, keepdims=True)
        return Xt


@adapt_fit_transform_docs
class Entropy(BaseEstimator, TransformerMixin):
    """Entropy of rows in a two-dimensional array.

    The rows of the array are interpreted as probability vectors,
    after taking absolute values if necessary and normalizing. Then,
    their Shannon entropies are computed and returned.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method exists to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each row in the array, take absolute values of any negative
        entry, normalise, and compute the Shannon entropy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, 1)
            Array of Shannon entropies.

        """
        # TODO: The following is a crude method to ensure each row vector
        #  consists of "probabilities" that sum to one. Consider normalisation
        #  in terms of bin counts?
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X)

        if np.any(Xt < 0):
            warnings.warn("Negative values detected in X! Taking absolute "
                          "value to calculate probabilities.")
            Xt = np.abs(Xt)

        Xt = Xt / Xt.sum(axis=1, keepdims=True)
        Xt = entr(Xt).sum(axis=1, keepdims=True) / np.log(2)
        return Xt


@adapt_fit_transform_docs
class Projection(BaseEstimator, TransformerMixin):
    """Projection onto specified columns.

    In practice, this simply means returning a selection of columns of the
    data.

    Parameters
    ----------
    columns : int or list of int, optional, default: ``0``
        The column indices of the array to project onto.

    """

    def __init__(self, columns=0):
        self.columns = columns

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method exists to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Return selected columns of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_columns)
            Output array, where ``n_columns = len(columns)``.

        """
        check_is_fitted(self, '_is_fitted')
        # Simple duck typing to handle case of pandas dataframe input
        if hasattr(X, 'columns'):
            # NB in this case we do not check the health of other columns
            Xt = check_array(X[self.columns], ensure_2d=False, copy=True)
        else:
            Xt = check_array(X, copy=True)
            Xt = Xt[:, self.columns]
        Xt = Xt.reshape(len(Xt), -1)
        return Xt
