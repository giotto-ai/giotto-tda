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

    def fit(self, XList, y = None):
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

        self.isFitted = True
        return self

    def transform(self, XList):
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
        check_is_fitted(self, ['isFitted'])

        XListTransformed = []

        maxOrder = max(self.orders)
        numberOrders = len(self.orders)

        lenXTransformed =  XList[0].shape[0] - maxOrder
        XTransformed = np.empty((lenXTransformed, numberOrders * XList[0].shape[1]))

        for i, order in enumerate(self.orders):
            XTransformed[:, i*XList[0].shape[1]:(i+1)*XList[0].shape[1]] = np.diff(XList[0], order, axis=0)[maxOrder-order:, :]

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1][maxOrder:, :])

        return XListTransformed
