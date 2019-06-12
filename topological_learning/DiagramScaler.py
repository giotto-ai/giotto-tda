import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.preprocessing as skprep
import sklearn.ensemble as skens
import math as m

import numpy as np

class DiagramScaler(BaseEstimator, TransformerMixin):
    """
    data sampling transformer that returns a sampled Pandas dataframe with a datetime index

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

    def __init__(self, scaler_kwargs={'scaler': skprep.MaxAbsScaler}):
        self.scaler_kwargs = scaler_kwargs

    def get_params(self, deep=True):
        return {'scaler_kwargs': self.scaler_kwargs}

    def _rotate_clockwise(X):
        rotationMatrix = m.sqrt(2) / 2. * np.array([[1, -1], [1 , 1]])
        return np.dot(X, rotationMatrix, axis=2)

    def _rotate_anticlockwise(X):
        rotationMatrix = m.sqrt(2) / 2. * np.array([[1, 1], [-1 , 1]])
        return np.dot(X, rotationMatrix, axis=2)

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
        scaler_kwargs = self.scaler_kwargs.copy()
        self.scaler = scaler_kwargs.pop('scaler')(**scaler_kwargs)

        X = self._rotate_anticlockwise(XList[0])
        self.scaler.fit(X[:, :, 1])

        self.isFitted = True
        return self

    #@jit
    def transform(self, XList, y = None, copy = None):
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

        XListScaled = []

        XScaled = self._rotate_anticlockwise(XList[0])
        XScaled[:, :, 1] = self.scaler.transform(XScaled[:, :, 1])
        XScaled = self._rotate_clockwise(XScaled)

        XListScaled.append(XScaled)

        if len(XList) == 2:
            XListScaled.append(XList[1])

        return XListScaled

    def inverse_transform(self, XList, copy = None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        # Check is fit had been called
        #check_is_fitted(self, ['isFitted'])

        XListScaled = []

        XScaled = self._rotate_anticlockwise(XList[0])
        XScaled[:, :, 1] = self.scaler.inverse_transform(XScaled[:, :, 1])
        XScaled = self._rotate_clockwise(XScaled)

        XListScaled.append(XScaled)

        if len(XList) == 2:
            XListScaled.append(XList[1])

        return XListScaled
