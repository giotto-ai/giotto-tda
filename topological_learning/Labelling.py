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

def derivation_function(function, X, deltaT=1, **function_kwargs):
    partialWindowBegin = function(X[:, deltaT:], axis=1, **function_kwargs)
    partialWindowEnd = function(X[:, :-deltaT], axis=1, **function_kwargs)
    derivative = (partialWindowEnd - partialWindowBegin) / partialWindowBegin / deltaT
    return derivative.reshape((-1, 1))

def variation_function(function, X, deltaT=1, **function_kwargs):
    fullWindow = function(X, axis=1, **function_kwargs)
    partialWindow = function(X[:, :-deltaT], axis=1, **function_kwargs)
    variation = (fullWindow - partialWindow) / partialWindow / deltaT
    return variation.reshape((-1, 1))

def application_function(function, X, axis=1, deltaT=0, **function_kwargs):
    return function(X, axis=1, **function_kwargs).reshape((-1, 1))

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

    implementedLabellingRecipes = {'application': application_function, 'variation': variation_function,
                                   'derivation': derivation_function}

    def __init__(self, labelling_kwargs={'type': 'derivation', 'deltaT': 1}, function_kwargs= {'function': np.std}, percentiles=None):
        self.labelling_kwargs = labelling_kwargs
        self.function_kwargs = function_kwargs
        self.percentiles = percentiles

    def get_params(self, deep=True):
        return {'labelling_kwargs': self.labelling_kwargs, 'function_kwargs': self.function_kwargs, 'percentiles': self.percentiles}

    @staticmethod
    def _validate_params(labelling_kwargs):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        if labelling_kwargs['type'] not in Labeller.implementedLabellingRecipes.keys():
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
        self._validate_params(self.labelling_kwargs)

        self.isFitted = True

        self._y = XList[1]

        labelling_kwargs = self.labelling_kwargs.copy()
        function_kwargs = self.function_kwargs.copy()
        self._yTransformed = Labeller.implementedLabellingRecipes[labelling_kwargs.pop('type')](function_kwargs.pop('function'), self._y, **labelling_kwargs, **function_kwargs)

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
            yTransformed = self._yTransformed
        else:
            labelling_kwargs = self.labelling_kwargs.copy()
            function_kwargs = self.function_kwargs.copy()
            yTransformed = Labeller.implementedLabellingRecipes[labelling_kwargs.pop('type')](function_kwargs.pop('function'), y, **labelling_kwargs, **function_kwargs)

        if self.thresholds is not None:
            yTransformedAbs = np.abs(yTransformed)
            yTransformed = np.concatenate([1 * (yTransformedAbs >= 0) * (yTransformedAbs < self.thresholds[0])] +\
                                          [1 * (yTransformedAbs >= self.thresholds[i])
                                           * (yTransformedAbs < self.thresholds[i+1])
                                           for i in range(len(self.thresholds)-1)] +\
                                          [1 * (yTransformedAbs >= self.thresholds[-1])], axis=1)
            yTransformed = np.nonzero(yTransformed)[1].reshape((y.shape[0], 1))

        XListTransformed = [ XList[0], yTransformed ]

        return XListTransformed
