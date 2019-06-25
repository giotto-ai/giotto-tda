import sklearn as sk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import numpy as np

class FeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Transformer that prepares the features and targets for the estimator.

    Parameters
    ----------
    samplingType : str
        The type of sampling

        - data_type: string, must equal either 'points' or 'distance_matrix'.
        - data_iter: an iterator. If data_iter is 'points' then each object in the iterator
          should be a numpy array of dimension (number of points, number of coordinates),
          or equivalent nested list structure. If data_iter is 'distance_matrix' then each
          object in the iterator should be a full (symmetric) square matrix (numpy array)
          of shape (number of points, number of points), __or a sparse distance matrix

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, n_steps_in_past=10):
        self.n_steps_in_past = n_steps_in_past

    def get_params(self, deep=True):
        return {'n_steps_in_past': self.n_steps_in_past}

    def fit(self, X, y = None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """

        return self

    def transform(self, X, y=None, copy=None):
        """ Implementation of the sk-learn transform function that samples the input.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        #check_is_fitted(self, ['isFitted'])

        n_samples = X.shape[0] - self.n_steps_in_past + 1
        indexer = np.arange(n_samples)[:, None] + np.arange(self.n_steps_in_past)[None, :]
        X_transformed = X[indexer]
        return X_transformed
