"""Time series labelling."""
# License: GNU AGPLv3

from numbers import Real
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, column_or_1d

from .embedding import SlidingWindow
from ..base import TransformerResamplerMixin
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class Labeller(BaseEstimator, TransformerResamplerMixin):
    """Target creation from sliding windows over a univariate time series.

    Useful to define a time series forecasting task in which labels are
    obtained from future values of the input time series, via the application
    of a function to time windows.

    Parameters
    ----------
    size : int, optional, default: ``10``
        Size of each sliding window.

    func : callable, optional, default: ``numpy.std``
        Function to be applied to each window.

    func_params : dict or None, optional, default: ``None``
        Additional keyword arguments for `func`.

    percentiles : list of real numbers between 0 and 100 inclusive, or \
        None, optional, default: ``None``
        If ``None``, creates a target for a regression task. Otherwise, creates
        a target for an n-class classification task where
        ``n = len(percentiles) + 1``.

    n_steps_future : int, optional, default: ``1``
        Number of steps in the future for the predictive task.

    Attributes
    ----------
    thresholds_ : list of floats or ``None`` if percentiles is ``None``
        Values corresponding to each percentile, based on data seen in
        :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.time_series import Labeller
    >>> # Create a time series
    >>> X = np.arange(10)
    >>> labeller = Labeller(size=3, func=np.min)
    >>> # Fit and transform X
    >>> X, y = labeller.fit_transform_resample(X, X)
    >>> print(X)
    [1 2 3 4 5 6 7 8]
    >>> print(y)
    [0 1 2 3 4 5 6 7]

    """

    _hyperparameters = {
        'size': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'func': {'type': Callable},
        'func_params': {'type': (dict, type(None))},
        'percentiles': {
            'type': (list, type(None)),
            'of': {'type': Real, 'in': Interval(0, 100, closed='both')}
            },
        'n_steps_future': {'type': int,
                           'in': Interval(1, np.inf, closed='left')}
        }

    def __init__(self, size=10, func=np.std,
                 func_params=None, percentiles=None, n_steps_future=1):
        self.size = size
        self.func = func
        self.func_params = func_params
        self.percentiles = percentiles
        self.n_steps_future = n_steps_future

    def fit(self, X, y=None):
        """Compute :attr:`thresholds_` and return the estimator.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, 1)
            Univariate time series to build a target for.

        y : None
            There is no need for a target, yet the pipeline API requires this
            parameter.

        Returns
        -------
        self : object

        """
        X = column_or_1d(X)
        validate_params(self.get_params(), self._hyperparameters)

        self._sliding_window = SlidingWindow(size=self.size, stride=1).fit(X)
        _X = self._sliding_window.transform(X)
        if self.func_params is None:
            self._effective_func_params = {}
        else:
            self._effective_func_params = self.func_params
        _X = self.func(_X, axis=1, **self._effective_func_params)[:, None]

        if self.percentiles is None:
            self.thresholds_ = None
        else:
            self.thresholds_ = [np.percentile(np.abs(_X.flatten()), percentile)
                                for percentile in self.percentiles]

        return self

    def transform(self, X, y=None):
        """Cuts `X` so it is aligned with `y`.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, 1)
            Univariate time series to build a target for.

        y : None
            There is no need for a target, yet the pipeline API requires this
            parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples_new,)
            The cut input time series.

        """
        check_is_fitted(self)
        Xt = column_or_1d(X)

        Xt = Xt[:-self.n_steps_future]

        if self.n_steps_future < self.size - 1:
            Xt = Xt[self.size - 1 - self.n_steps_future:]
        return Xt

    def resample(self, y, X=None):
        """Resample `y`.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Time series to build a target for.

        X : None
            There is no need for `X`, yet the pipeline API requires this
            parameter.

        Returns
        -------
        yr : ndarray of shape (n_samples_new,)
            Target for the prediction task.

        """
        check_is_fitted(self)
        y = column_or_1d(y)

        yr = self._sliding_window.transform(y)
        yr = self.func(yr, axis=1, **self._effective_func_params)[:, None]

        if self.thresholds_ is not None:
            yr = np.abs(yr)
            yr = np.concatenate(
                [1 * (yr >= 0) * (yr < self.thresholds_[0])] +
                [1 * (yr >= self.thresholds_[i]) *
                 (yr < self.thresholds_[i + 1]) for i in range(
                    len(self.thresholds_) - 1)] +
                [1 * (yr >= self.thresholds_[-1])], axis=1)
            yr = np.nonzero(yr)[1].reshape(yr.shape[0], 1)

        if self.n_steps_future > self.size - 1:
            yr = yr[self.n_steps_future - self.size + 1:]

        return yr.reshape(-1)
