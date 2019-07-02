import sklearn as sk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import numpy as np

class FeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Transformer that rearanges the features in sequences of features for the
    final estimator.

    Parameters
    ----------
    n_steps_in_past : int
        Number of previous time steps to be used for sequence-based predictions.

    is_keras : boolean
        Whether the final estimator is a neural_network.
    """

    def __init__(self, n_steps_in_past=10, is_keras=False):
        self.n_steps_in_past = n_steps_in_past
        self.is_keras = is_keras

    def get_params(self, deep=True):
        return {'n_steps_in_past': self.n_steps_in_past, 'is_keras': self.is_keras}

    @staticmethod
    def _validate_params():
        """
        A class method that checks whether the hyperparameters and the input parameters
        of the :meth:'fit' are valid.
        """
        pass

    def fit(self, X, y = None):
        """
        Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_featurers)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params()

        self._is_fitted = True
        return self

    def transform(self, X, y=None, copy=None):
        """
        Rearange input features X into a sequence of n_steps_in_past features.
        If is_keras = False, the sequence of features is unrolled.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples-n_steps_in_past+1, n_steps_in_past)
        if is_keras=True or (n_samples-n_steps_in_past+1, n_features*n_steps_in_past) else
            Rearanged features array by sequences of n_steps_in_past.
        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_itted'])

        n_samples = X.shape[0] - self.n_steps_in_past + 1
        indexer = np.arange(n_samples)[:, None] + np.arange(self.n_steps_in_past)[None, :]
        X_transformed = X[indexer]

        if not self.is_keras:
            X_transformed = X_transformed.reshape((n_samples, -1))

        return X_transformed
