# License : Apache 2.0
import types

import numpy as np
import numbers
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, column_or_1d
from ..base import TransformerResamplerMixin
from .embedding import SlidingWindow
from ..utils.validation import validate_params


def _derivation_function(function, X, delta=1, **function_params):
    partial_window_begin = function(X[:, :-delta], axis=1, **function_params)
    partial_window_end = function(X[:, delta:], axis=1, **function_params)
    duration_ = (partial_window_end - partial_window_begin)
    derivative = duration_ / partial_window_begin / delta
    derivative[(partial_window_begin == 0) & (partial_window_end == 0)] = 0
    return derivative.reshape((-1, 1))


def _variation_function(function, X, delta=1, **function_params):
    full_window = function(X, axis=1, **function_params)
    partial_window = function(X[:, :-delta], axis=1, **function_params)
    variation = (full_window - partial_window) / partial_window / delta
    variation[(partial_window == 0) & (full_window == 0)] = 0
    return variation.reshape((-1, 1))


def _application_function(function, X, delta=0, **function_params):
    return function(X, axis=1, **function_params).reshape((-1, 1))


class Labeller(BaseEstimator, TransformerResamplerMixin):
    """
    Target transformer.

    Parameters
    ----------
    samplingType : str
        The type of sampling

        - data_type: string, must equal either 'points' or 'distance_matrix'.
        - data_iter: an iterator. If data_iter is 'points' then each object
          in the iterator should be a numpy array of dimension (number of
          points, number of coordinates), or equivalent nested list structure.
          If data_iter is 'distance_matrix' then each object in the iterator
          should be a full (symmetric) square matrix (numpy array) of shape (
          number of points, number of points), __or a sparse distance matrix

    Attributes
    ----------
    is_fitted : boolean
        Whether the transformer has been fitted
    """
    implemented_labelling_recipes = {'application': _application_function,
                                     'variation': _variation_function,
                                     'derivation': _derivation_function}
    _hyperparameters = {
        'labelling':
            (str, ['application', 'variation', 'derivation']),
        'delta': [int, (1, np.inf)],
        'function': [types.FunctionType],
        'percentiles': (list, [numbers.Number, (0., 1.)]),
        'n_steps_future': (int, [1, np.inf])}

    def __init__(self, width=2, stride=1, labelling='application', delta=1,
                 function=np.std, function_params=None, percentiles=None,
                 n_steps_future=1):
        self.width = width
        self.stride = stride
        self.labelling = labelling
        self.delta = delta
        self.function = function
        self.function_params = function_params
        self.percentiles = percentiles
        self.n_steps_future = n_steps_future

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Attributes
        __________
        thresholds_ :

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        column_or_1d(X)

        if self.function_params is None:
            self.effective_function_params_ = {}
        else:
            self.effective_function_params_ = self.function_params.copy()

        self._labeller = self.implemented_labelling_recipes[self.labelling]

        self._sliding_window = SlidingWindow(width=self.width,
                                             stride=self.stride).fit(X)

        _X = self._sliding_window.transform(X)
        _X = self._labeller(self.function, _X, self.delta,
                            **self.effective_function_params_)

        if self.percentiles is not None:
            self.thresholds_ = [
                np.percentile(np.abs(_X.flatten()), percentile)
                for percentile in self.percentiles]
        else:
            self.thresholds_ = None
        return self

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data. ``

        y : None
            There is no need of a target, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples_new, n_features)
            The transformed/resampled input array.
            ``n_samples_new = n_samples // period``.

        """
        Xt = column_or_1d(X).copy()

        Xt = Xt[:-self.n_steps_future]

        if self.n_steps_future < self.width:
            Xt = Xt[self.width - 1 - self.n_steps_future:]
        return Xt

    def resample(self, y, X=None):
        """Resample y.

        Parameters
        ----------
        y : ndarray, shape (n_samples, n_features)
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target.
            ``n_samples_new = n_samples - 1``.

        """
        # Check is fit had been called
        check_is_fitted(self, ['_labeller', '_sliding_window', 'threshold_'])
        column_or_1d(y)

        yt = self._sliding_window.transform(y)
        yt = self._labeller(self.function, yt, self.delta,
                            **self.effective_function_params_)

        if self.thresholds_ is not None:
            yt = np.abs(yt)
            yt = np.concatenate(
                [1 * (yt >= 0) * (yt < self.thresholds_[0])] +
                [1 * (yt >= self.thresholds_[i]) *
                 (yt < self.thresholds_[i + 1]) for i in range(
                    len(self.thresholds_) - 1)] +
                [1 * (yt >= self.thresholds_[-1])], axis=1)
            yt = np.nonzero(yt)[1].reshape((y.shape[0], 1))

        if self.n_steps_future >= self.width:
            yt = yt[self.n_steps_future - self.width + 1:]

        return yt
