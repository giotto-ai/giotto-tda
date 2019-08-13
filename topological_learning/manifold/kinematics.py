# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: TBD

import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class Kinematics(BaseEstimator, TransformerMixin):
    """
    Transform a time series of points in an embedded space into a time
    series of derivatives of those points.

    Parameters
    ----------
    orders : list of ints, default: [0, 1, 2]
        List of derivative orders to return.

    Examples
    --------
    >>> from topological_learning.manifold import StatefulMDS, Derivatives
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = MDS(n_components=2)
    >>> X_embedded = embedding.fit(X[:100]).transform(X[:100])
    >>> derivatives = Derivatives(orders=[0, 1, 2])
    >>> X_derived = derivatives.fit(X_embedded).transform(X_embedded)
    >>> X_derived.shape
    (98, 2)
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
        """Do nothing and return the estimator unchanged.
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

    def transform(self, X):
        """Computes the position of the points X in the same embedding space calculated
        in fit and returns the embedded coordinates

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples - max_order, n_features*n_orders)
            Points and their derivative at the required orders where max_order is the
            maximum values of orders and n_orders is the length of orders.
        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        max_order = max(self.orders)
        n_orders = len(self.orders)

        n_samples_transformed =  X.shape[0] - max_order
        X_transformed = np.empty((n_samples_transformed, n_orders * X.shape[1]))

        for i, order in enumerate(self.orders):
            X_transformed[:, i*X.shape[1]:(i+1)*X.shape[1]] = np.diff(X, order, axis=0)[max_order-order:, :]

        return X_transformed
