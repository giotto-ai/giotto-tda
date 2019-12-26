import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import entr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class Eccentricity(BaseEstimator, TransformerMixin):
    """Maps dataset to reals using the eccentricity filter function.

    Parameters
    ----------
    exponent : int or np.inf, default: `np.inf`
        The exponent used to calculate the eccentricity.

    metric : str or function, default: `'euclidean'`
        The distance metric to use. If a string, this may be one of the metrics
        supported by scipy.spatial.distance.pdist

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
        X : ndarray of sparse or dense arrays, shape (n_samples,)
            Input data.

        y : None
            There is no need for a target in fit, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X)
        if self.metric_params is None:
            self.effective_metric_params_ = dict()
        else:
            self.effective_metric_params_ = self.metric_params.copy()
        return self

    def transform(self, X, y=None):
        """Apply the eccentricity filter function to each row in the distance
        matrix derived from `X`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, 1)

        """
        check_is_fitted(self)
        X = check_array(X)
        if self.metric == 'precomputed':
            Xt = X
        else:
            Xt = squareform(
                pdist(X, metric=self.metric, **self.effective_metric_params_))
        Xt = np.linalg.norm(Xt, axis=1, ord=self.exponent).reshape(-1, 1)
        return Xt


class Entropy(BaseEstimator, TransformerMixin):
    """Maps dataset to reals using the entropy filter function.

    """
    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method exists to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of sparse or dense arrays, shape (n_samples,)
            Input data.

        y : None
            There is no need for a target in fit, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Normalise each row in array to have unit norm and calculate the
        Shannon entropy.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, 1)

        """
        # TODO: the following is a crude method to ensure each row vector
        #  consists of "probabilities" that sum to one. Consider normalisation
        #  in terms of bin counts?
        check_is_fitted(self)
        X = check_array(X)

        if np.any(X < 0):
            warnings.warn("Negative values detected in X! Taking absolute "
                          "value to calculate probabilities.")
            X = np.abs(X)

        probs = X / X.sum(axis=1, keepdims=True)
        Xt = (entr(probs).sum(axis=1) / np.log(2)).reshape(-1, 1)
        return Xt


class Projection(BaseEstimator, TransformerMixin):
    """Maps dataset to reals by projecting onto specified column.

    Parameters
    ----------
    column_indices : int or list of ints, default: `0`
                     The column indices of the array to project onto.

    """
    def __init__(self, column_indices=0):
        self.column_indices = column_indices

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method exists to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of sparse or dense arrays, shape (n_samples,)
            Input data.

        y : None
            There is no need for a target in fit, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Fit and project the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, 1)

        """
        check_is_fitted(self)
        X = check_array(X)
        Xt = X[:, self.column_indices].reshape(X.shape[0], -1)
        return Xt
