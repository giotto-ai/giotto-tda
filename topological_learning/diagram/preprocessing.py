import math as m
import numpy as np
import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from sklearn.utils._joblib import Parallel, delayed
import itertools

from ._utils import _sort, _filter
from ._metrics import _parallel_norm
from .distance import DiagramDistance


class DiagramStacker(BaseEstimator, TransformerMixin):
    """
    Transformer for the calculation of persistence diagrams from Vietoris-Rips filtration.

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
    is_fitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def fit(self, X, y=None):
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

    def transform(self, X, y=None):
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

        X_transformed = { None: np.concatenate(list(X.values()), axis=1)}
        return X_transformed


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
    is_fitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, norm_kwargs={'norm': 'bottleneck', 'order': np.inf}, function=np.max, n_jobs=1):
        self.norm_kwargs = norm_kwargs
        self.function = function
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'norm_kwargs': self.norm_kwargs, 'function': self.function, 'n_jobs': self.n_jobs}

    def fit(self, X, y=None):
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
        norm_kwargs = self.norm_kwargs.copy()

        norm_array = _parallel_norm(X, norm_kwargs, self.n_jobs)
        self._scale = self.function(norm_array)

        self.is_fitted = True
        return self

    #@jit
    def transform(self, X, y=None):
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
        check_is_fitted(self, ['is_fitted'])

        X_scaled = { dimension: X / self._scale for dimension, X in X.items() }
        return X_scaled

    def inverse_transform(self, X, copy=None):
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
        check_is_fitted(self, ['is_fitted'])

        X_scaled = { dimension: X * self._scale for dimension, X in X.items() }
        return X_scaled

class DiagramFilter(BaseEstimator, TransformerMixin):
    implemented_filtering_parameters_types = ['fixed', 'search']

    def __init__(self, homology_dimensions=None, filtering_parameters_type='search', delta=0.,
                 metric_kwargs={'metric': 'bottleneck', 'order': np.inf}, epsilon=1.,
                 tolerance=1e-2, max_iteration=20, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.filtering_parameters_type = filtering_parameters_type
        self.delta = delta
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'homology_dimensions': self.homology_dimensions,
                'filtering_parameters_type': self.filtering_parameters_type,
                'delta': self.delta,
                'metric_kwargs': self.metric_kwargs,
                'epsilon': self.epsilon,
                'tolerance': self.tolerance,
                'max_iteration': self.max_iteration,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(filtering_parameters_type):
        if filtering_parameters_type not in DiagramFilter.implemented_filtering_parameters_types:
            raise ValueError('The filtering parameters type you specified is not implemented')

    def _bisection(self, X):
        iterator = iter([(i, i) for i in range(len(X))])

        numberPoints = [ X[dimension].shape for dimension in self.homologyDimensions ]
        XConcatenated = np.concatenate([X[dimension] for dimension in self.homologyDimensions])

        lowerCutoff = 0.
        upperCutoff = 1.

        currentMeanDistance = 0.
        previousMeanDistance = 1.

        while m.abs(currentMeanDistance - previousMeanDistance) > self.tolerance and iteration <= self.max_iteration:
            middlePoint = (lowerPoint + upperPoint) // 2.
            middlePointIndex = indices[:, middlePoint]
            # cutoff = m.sqrt(2.)/2. * (XConcatenated[indices[middlePoint][0], indices[middlePoint][0], 1] - XConcatenated[, , 0])
            # XFiltered = _filter(XConcatenated, self.homologyDimensions, middleCutoff)

            # XCOncatenated and XFIltered need to have the same homology dimensions!!!!!
            # distance = _parallel_pairwise(XConcatenated, XFiltered, iterator, self.n_jobs)

            if distance == tolerance:
                return middleCutoff
            elif (distance - tolerance)*() < 0:
                upperCutoff = middleCutoff
            else:
                lowerCutoff = middleCutoff

        return middleCutoff

    def fit(self, X, y=None):
        if not self.homology_dimensions:
            self.homology_dimensions = set(X.keys())

        self._validate_params(self.filtering_parameters_type)

        # self.delta = self.delta

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted'])

        X = _sort(X, self.homology_dimensions)

        X_filtered = _filter(X, self.homology_dimensions, self.delta)
        return X_filtered
