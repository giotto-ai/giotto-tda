from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import pdist, squareform
import numpy as np


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

    def __init__(self, exponent=np.inf, metric='euclidean', metric_params=None):
        self.exponent = exponent
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else dict()

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
        return self

    def transform(self, X, y=None):
        """Apply the eccentricity filter function to each row in the distance matrix derived from `X`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples,)
        """
        distance_matrix = squareform(
            pdist(X, metric=self.metric, **self.metric_params)
        )
        Xt = np.linalg.norm(distance_matrix, axis=1, ord=self.exponent)
        return Xt
