# License: Apache 2.0

import numpy as np

from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class GraphEntropy(BaseEstimator, TransformerMixin):
    """Compute diffusion entropy.

    Once the diffusion has been computed, the related entropy at
    each point in time is calculated. The entropy values will be
    used as features to perform the embedding. In order to compute
    the entropy the absolute values of the diffusion vectors are taken.
    In order to preserve the invariance of the entropy calculation with
    respect to the orientation of the simplices in the complex the
    initial condition has to be a delta (all energy placed in one simplex).

    """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray
            Input containing the diffusion vectors for n_simplices with
            different initial condition and times, shape
            (n_simplices, n_initial_conditions, n_times).

        y : None
            There is no need for a target in this fit, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, allow_nd=True)
        self.is_fitted_ = True

        return self

    def transform(self, X, y=None):
        """Compute Entropy for diffusion vectors.

        Parameters
        ----------
        X : ndarray
            Input containing the diffusion vectors for n_simplices with
            different initial condition and times, shape
            (n_simplices, n_initial_conditions, n_times).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        entropies : ndarray
            Entropies of diffusion vectors, shape (n_simplices, n_times).
        """
        check_is_fitted(self, ['is_fitted_'])
        check_array(X, allow_nd=True)
        entropies = entropy(np.abs(X), base=2)

        return np.nan_to_num(entropies, 0)
