"""Resampling and stationarization of time series data."""
# License: GNU AGPLv3

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted

from ..base import TransformerResamplerMixin
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class Resampler(BaseEstimator, TransformerResamplerMixin):
    """Time series resampling at regular intervals.

    Parameters
    ----------
    period : int, default: ``2``
        The sampling period, i.e. one point every period will be kept.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.time_series import Resampler
    >>> # Create a noisy signal
    >>> signal = np.asarray([np.sin(x /40) + np.random.random()
    ...                      for x in range(0, 300)])
    >>> # Set up the Resampler
    >>> period = 10
    >>> periodic_sampler = Resampler(period=period)
    >>> # Fit and transform the signal
    >>> signal_resampled = periodic_sampler.fit_transform(signal)
    >>> print(signal_resampled.shape)
    (30,)

    """

    _hyperparameters = {
        'period': {'type': int, 'in': Interval(1, np.inf, closed='left')}
        }

    def __init__(self, period=2):
        self.period = period

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, ...)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        check_array(X, ensure_2d=False, allow_nd=True)
        validate_params(self.get_params(), self._hyperparameters)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Resample `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, ...)
            Input data.

        y : None
            There is no need for a target, yet the pipeline API requires this
            parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples_new, ...)
            Resampled array. ``n_samples_new = n_samples // period``.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X, ensure_2d=False, allow_nd=True, copy=True)

        if Xt.ndim == 1:
            Xt = Xt[: None]
        Xt = Xt[::self.period]

        return Xt

    def resample(self, y, X=None):
        """Resample `y`.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target.

        X : None
            There is no need for input data, yet the pipeline API requires this
            parameter.

        Returns
        -------
        yr : ndarray of shape (n_samples_new,)
            Resampled target. ``n_samples_new = n_samples // period``.

        """
        check_is_fitted(self, '_is_fitted')
        yr = column_or_1d(y)
        yr = yr[::self.period]

        return yr


class Stationarizer(BaseEstimator, TransformerResamplerMixin):
    """Methods for stationarizing time series data.

    Time series may be stationarized to remove or reduce linear or exponential
    trends.

    Parameters
    ----------
    operation : ``'return'`` | ``'log-return'``, default: ``'return'``
        The type of stationarization operation to perform. It can have two
        values:

        - ``'return'``:
          This option transforms the time series :math:`{X_t}_t` into the
          time series of relative returns, i.e. the ratio :math:`(X_t-X_{
          t-1})/X_t`.

        - ``'log-return'``:
          This option transforms the time series :math:`{X_t}_t` into the
          time series of relative log-returns, i.e. :math:`\\log(X_t/X_{
          t-1})`.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.time_series import Stationarizer
    >>> # Create a noisy signal
    >>> signal = np.asarray([np.sin(x /40) + 5 + np.random.random()
    >>>                      for x in range(0, 300)]).reshape(-1, 1)
    >>> # Initialize the stationarizer
    >>> stationarizer = Stationarizer(operation='return')
    >>> # Fit and transform the signal
    >>> signal_stationarized = stationarizer.fit_transform(signal)
    >>> print(signal_stationarized.shape)
    (299,)

    """

    _hyperparameters = {
        'operation': {'type': str, 'in': ['return', 'log-return']}
        }

    def __init__(self, operation='return'):
        self.operation = operation

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, ...)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        check_array(X, ensure_2d=False, allow_nd=True)
        validate_params(self.get_params(), self._hyperparameters)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Stationarize `X` by applying the procedure given by `operation`.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, ...)
            Input data.

        y : None
            There is no need for a target, yet the pipeline API requires this
            parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples_new, ...)
            Stationarized array. ``n_samples_new = n_samples - 1``.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X, ensure_2d=False, allow_nd=True)

        if Xt.ndim == 1:
            Xt = Xt[:, None]

        if self.operation == 'return':
            return np.diff(Xt, n=1, axis=0) / Xt[1:]
        else:  # Assumes 'log-return' operation
            return np.diff(np.log(Xt), n=1, axis=0)

    def resample(self, y, X=None):
        """Resample `y`.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target.

        X : None
            There is no need for input data, yet the pipeline API requires this
            parameter.

        Returns
        -------
        yr : ndarray of shape (n_samples_new,)
            Resampled target. ``n_samples_new = n_samples - 1``.

        """
        check_is_fitted(self, '_is_fitted')
        y = column_or_1d(y)

        return y[1:]
