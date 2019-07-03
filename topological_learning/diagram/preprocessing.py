import math as m
import numpy as np
import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from sklearn.utils._joblib import Parallel, delayed
import itertools

from ._utils import _sort, _filter, _sample
from ._metrics import _parallel_norm
from .distance import DiagramDistance


class DiagramStacker(BaseEstimator, TransformerMixin):
    """Transformer for stacking persistence subdiagrams. Useful when topological
    persistence information per sample has been previously separated according
    to some criterion (e.g. by homology dimension if produced by an instance of
    ```VietorisRipsPersistence``).

    """

    def __init__(self):
        pass

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default: True
            Behaviour not yet implemented.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return {}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers
            d representing homology dimensions, and whose values are ndarrays of
            shape (n_samples, M_d, 2) whose each entries along axis 0 are persistence
            diagrams with M_d persistent topological features. For example, X
            could be the result of applying the ``transform`` method of a
            ``VietorisRipsPersistence`` transformer to a collection of point
            clouds/distance matrices, but only if that transformer was instantiated
            with ``pad=True``.

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

    def transform(self, X, y=None):
        """Stacks all available persistence subdiagrams corresponding to each sample
        into one persistence diagram.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers
            d representing homology dimensions, and whose values are ndarrays of
            shape (n_samples, M_d, 2) whose each entries along axis 0 are persistence
            diagrams with M_d persistent topological features. For example, X
            could be the result of applying the ``transform`` method of a
            ``VietorisRipsPersistence`` transformer to a collection of point
            clouds/distance matrices, but only if that transformer was instantiated
            with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : dict of None: ndarray
            Dictionary with a single ``None`` key, and corresponding value an
            ndarray of shape (n_samples, :math:`\\sum_{\\mathrm{d}}` M_d, 2).

        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

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

    """

    def __init__(self, norm='bottleneck', norm_params={'order': np.inf}, function=np.max, n_jobs=None):
        self.norm = norm
        self.norm_params = norm_params
        self.function = function
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'norm': self.norm, 'norm_params': self.norm_params, 'function': self.function, 'n_jobs': self.n_jobs}

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
        norm_params = self.norm_params.copy()

        sampling = { dimension: None for dimension in X.keys() }

        if self.norm in ['landscape', 'betti']:
            n_samples = norm_params.pop('n_samples')
            norm_params['sampling'] = _sample(X, n_samples)

        norm_array = _parallel_norm(X, self.norm, norm_params, self.n_jobs)
        self._scale = self.function(norm_array)

        self._is_fitted = True
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
        check_is_fitted(self, ['_is_fitted'])

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
        check_is_fitted(self, ['_is_fitted'])

        X_scaled = { dimension: X * self._scale for dimension, X in X.items() }
        return X_scaled

class DiagramFilter(BaseEstimator, TransformerMixin):
    implemented_filtering_parameters_types = ['fixed', 'search']

    def __init__(self, homology_dimensions=None, filtering_parameters_type='search', delta=0.,
                 metric='bottleneck', metric_params={'order': np.inf}, epsilon=1.,
                 tolerance=1e-2, max_iteration=20, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.filtering_parameters_type = filtering_parameters_type
        self.delta = delta
        self.metric = metric
        self.metric_params = metric_params
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'homology_dimensions': self.homology_dimensions,
                'filtering_parameters_type': self.filtering_parameters_type,
                'delta': self.delta,
                'metric': self.metric,
                'metric_params': self.metric_params,
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

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X = _sort(X, self.homology_dimensions)

        X_filtered = _filter(X, self.homology_dimensions, self.delta)
        return X_filtered
