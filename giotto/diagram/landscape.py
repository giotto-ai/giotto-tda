# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from giotto.utils.validation import check_diagram
from giotto.diagram._metrics import landscape_function
from giotto.diagram._utils import _discretize


class PersistenceLandscape(BaseEstimator, TransformerMixin):
    """
    DOC TO DO
    """
    def __init__(self, n_layers=1, n_sampled_values=100, n_jobs=None):
        self.n_layers = n_layers
        self.n_sampled_values = n_sampled_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """DOC TO DO

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers representing
            homology dimensions, and whose values are ndarrays of shape (n_samples, M, 2)
            whose each entries along axis 0 are persistence diagrams.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """

        X = check_diagram(X)

        self._linspaces, _ = \
            _discretize(X, n_sampled_values=self.n_sampled_values)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """TO DO

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers representing
            homology dimensions, and whose values are ndarrays of shape (n_samples, M, 2)
            whose each entries along axis 0 are persistence diagrams.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_samples)
            Amplitude of the diagrams in X.

        """
        check_is_fitted(self, ['_is_fitted'])
        X = check_diagram(X)

        n_samples = len(next(iter(X.values())))

        # Only parallelism is across dimensions
        pls = Parallel(n_jobs=n_jobs)(delayed(landscape_function)(
            X[dim], self._linspaces[dim][:, None, None]) for dim in X.keys())
        pls = np.stack(pls, axis=1)
        linspaces = np.vstack([self._linspaces[dim] for dim in X.keys()])
        linspaces = np.tile(linspaces, (n_samples, 1, 1, 1))
        X_transformed = np.stack([linspaces, pls], axis=3)
        # Shape: (n_samples, n_homology_dimensions, n_layers, 2,
        # n_sampled_values)
        return X_transformed
