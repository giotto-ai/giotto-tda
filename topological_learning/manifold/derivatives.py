import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class Derivatives(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, orders=[0, 1, 2]):
        self.orders = orders

    def get_params(self, deep=True):
        return {'orders': self.orders}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

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
        self._validate_params()

        self.is_fitted = True
        return self

    def transform(self, X):
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
        check_is_fitted(self, ['is_fitted'])

        max_order = max(self.orders)
        n_orders = len(self.orders)

        n_samples_transformed =  X.shape[0] - max_order
        X_transformed = np.empty((n_samples_transformed, n_orders * X.shape[1]))

        for i, order in enumerate(self.orders):
            X_transformed[:, i*X.shape[1]:(i+1)*X.shape[1]] = np.diff(X, order, axis=0)[max_order-order:, :]

        return X_transformed
