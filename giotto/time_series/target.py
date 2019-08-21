import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


def _derivation_function(function, X, delta_t=1, **function_kwargs):
    partial_window_begin = function(X[:, delta_t:], axis=1, **function_kwargs)
    partial_window_end = function(X[:, :-delta_t], axis=1, **function_kwargs)
    derivative = (partial_window_end - partial_window_begin) / \
        partial_window_begin / delta_t
    derivative[(partial_window_begin == 0) & (partial_window_end == 0)] = 0
    return derivative.reshape((-1, 1))


def _variation_function(function, X, delta_t=1, **function_kwargs):
    full_window = function(X, axis=1, **function_kwargs)
    partial_window = function(X[:, :-delta_t], axis=1, **function_kwargs)
    variation = (full_window - partial_window) / partial_window / delta_t
    variation[(partial_window == 0) & (full_window == 0)] = 0
    return variation.reshape((-1, 1))


def _application_function(function, X, axis=1, delta_t=0, **function_kwargs):
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
    is_fitted : boolean
        Whether the transformer has been fitted
    """
    implementedLabellingRecipes = {'application': _application_function,
                                   'variation': _variation_function,
                                   'derivation': _derivation_function}

    def __init__(self, labelling_kwargs={'type': 'derivation', 'delta_t': 1},
                 function_kwargs={'function': np.std},
                 window_size=2, percentiles=None, n_steps_future=1):
        self.labelling_kwargs = labelling_kwargs
        self.function_kwargs = function_kwargs
        self.window_size = window_size
        self.percentiles = percentiles
        self.n_steps_future = n_steps_future

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
        return {'labelling_kwargs': self.labelling_kwargs,
                'function_kwargs': self.function_kwargs,
                'window_size': self.window_size,
                'percentiles': self.percentiles,
                'n_steps_future': self.n_steps_future}

    @staticmethod
    def _embed(y, window_size):
        n_windows = y.shape[0] - window_size + 1

        y = np.flip(y)
        y_embedded = np.stack(
            [y[i: i + window_size].flatten() for i in range(0, n_windows)])

        return np.flip(y_embedded).reshape((n_windows, window_size))

    @staticmethod
    def _roll(y, n_steps_future):
        return y[n_steps_future:]

    @staticmethod
    def _validate_params(labelling_kwargs):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:'fit' are valid.
        """
        if labelling_kwargs['type'] not in \
                Labeller.implementedLabellingRecipes.keys():
            raise ValueError(
                'The labelling type you specified is not implemented')

    def fit(self, y, X=None):
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

        self._y = self._embed(y, self.window_size)

        labelling_kwargs = self.labelling_kwargs.copy()
        function_kwargs = self.function_kwargs.copy()
        self._y_transformed = Labeller.implementedLabellingRecipes[
            labelling_kwargs.pop('type')](function_kwargs.pop('function'),
                                          self._y, **labelling_kwargs,
                                          **function_kwargs)

        if self.percentiles is not None:
            self.thresholds = [
                np.percentile(np.abs(self._y_transformed.flatten()),
                              percentile) for percentile in self.percentiles]
        else:
            self.thresholds = None

        self.is_fitted = True
        return self

    def transform(self, y):
        """Implementation of the sk-learn transform function that samples the input.

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

        y = self._embed(y, self.window_size)
        if np.array_equal(y, self._y):
            y_transformed = self._y_transformed
        else:
            labelling_kwargs = self.labelling_kwargs.copy()
            function_kwargs = self.function_kwargs.copy()
            y_transformed = Labeller.implementedLabellingRecipes[
                labelling_kwargs.pop('type')](function_kwargs.pop('function'),
                                              y, **labelling_kwargs,
                                              **function_kwargs)

        # FIXME: simplify
        if self.thresholds is not None:
            y_transformedAbs = np.abs(y_transformed)
            y_transformed = np.concatenate(
                [1 * (y_transformedAbs >= 0) * (y_transformedAbs <
                                                self.thresholds[0])] +
                [1 * (y_transformedAbs >= self.thresholds[i]) *
                 (y_transformedAbs < self.thresholds[i + 1]) for i in range(
                    len(self.thresholds) - 1)] +
                [1 * (y_transformedAbs >= self.thresholds[-1])], axis=1)
            y_transformed = np.nonzero(y_transformed)[1].reshape(
                (y.shape[0], 1))

        y_transformed = self._roll(y_transformed, self.n_steps_future)
        return y_transformed

    def cut(self, X):
        X_cut = X[self.window_size - self.n_steps_future:-self.n_steps_future]
        return X_cut
