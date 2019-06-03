import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class LorenzLabeller(BaseEstimator, TransformerMixin):
    """
    Target transformer for the Lorenz attractor.

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

    def __init__(self):
        pass

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

        y = XList[2]

        yTransformed = y.reshape((-1, y.shape[2] // 2, 2))
        yTransformed = np.mean(yTransformed, axis=1)
        XListTransformed = [XList[0], yTransformed]

        return XListTransformed

def derivation_function(function, X, axis=1, deltaT=1):
    partialWindowBegin = function(X[:, deltaT:], axis=axis)
    partialWindowEnd = function(X[:, :-deltaT], axis=axis)
    return (partialWindowEnd - partialWindowBegin) / partialWindowBegin / deltaT

def variation_function(function, X, axis=1, deltaT=1):
    fullWindow = function(X, axis=axis)
    partialWindow = function(X[:, :-deltaT], axis=axis)
    return (fullWindow - partialWindow) / partialWindow / deltaT

def apply_function(function, X, axis=1, deltaT=0):
    return function(X, axis=axis)

class Labeller(BaseEstimator, TransformerMixin):
    """
    Target transformer.

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

    implementedLabellingRecipes = {'apply': apply_function, 'variation': variation_function,
                                   'derivation': derivation_function}

    def __init__(self, labellingType='derivation', function=np.std, deltaT=0, percentiles=None, **function_kwargs):
        self.labellingType = labellingType
        self.function = function
        self.deltaT = deltaT
        self.percentiles = percentiles
        self.function_kwargs = function_kwargs

    def get_params(self, deep=True):
        return {'labellingType': self.labellingType,'function': self.function, 'deltaT': self.deltaT,
                'percentiles': self.percentiles}

    @staticmethod
    def _validate_params(labellingType):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        if labellingType not in Labeller.implementedLabellingRecipes.keys():
            raise ValueError('The labelling type you specified is not implemented')

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
        self._validate_params(self.labellingType)

        self.isFitted = True

        self._y = XList[1]
        y = self._y.reshape((self._y.shape[0], -1))

        self._yTransformed = Labeller.implementedLabellingRecipes[self.labellingType](self.function, y, axis=1, deltaT=self.deltaT, **self.function_kwargs)

        if self.percentiles is not None:
            self.thresholds = [np.percentile(np.abs(self._yTransformed.flatten()), percentile) for percentile in self.percentiles]
        else:
            self.thresholds = None

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

        y = XList[1]

        if np.array_equal(y, self._y):
            yTransformed = self._yTransformed.reshape((y.shape[0], -1))
        else:
            y = y.reshape((y.shape[0], -1))

            yTransformed = Labeller.implementedLabellingRecipes[self.labellingType](self.function, y, axis=1, deltaT=self.deltaT, **self.function_kwargs).reshape((y.shape[0], -1))

        if self.thresholds is not None:
            yTransformedAbs = np.abs(yTransformed)
            yTransformed = np.concatenate([1 * (yTransformedAbs >= 0) * (yTransformedAbs < self.thresholds[0])] +\
                                          [1 * (yTransformedAbs >= self.thresholds[i])
                                           * (yTransformedAbs < self.thresholds[i+1])
                                           for i in range(len(self.thresholds)-1)] +\
                                          [1 * (yTransformedAbs >= self.thresholds[-1])], axis=1)
            yTransformed = np.nonzero(yTransformed)[1]

        XListTransformed = [ XList[0], yTransformed.reshape((y.shape[0], 1)) ]

        return XListTransformed
