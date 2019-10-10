# License: Apache 2.0

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from ..base import TransformerResamplerMixin
from ..utils.validation import validate_params
from sklearn.utils.validation import check_array, column_or_1d
import numpy as np


class Resampler(BaseEstimator, TransformerResamplerMixin):
    """Data sampling transformer that returns a sampled numpy.ndarray.

    Parameters
    ----------
    period : int, default: 2
        The sampling period, i.e. pointd every period will be kept.


    Attributes
    ----------
    _n_features : int
        Number of features (i.e. number fo time series) passed as an input
        of the resampler.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from giotto.time_series import Resampler
    >>> # Create a noisy signal sampled
    >>> signal = np.asarray([np.sin(x /40) + np.random.random()
    ... for x in range(0, 300)])
    >>> plt.plot(signal)
    >>> plt.show()
    >>> # Set up the Resampler
    >>> period = 10
    >>> periodic_sampler = Resampler(period=period)
    >>> # Fit and transform the DataFrame
    >>> periodic_sampler.fit(signal)
    >>> signal_resampled = periodic_sampler.transform(signal)
    >>> plt.plot(signal_resampled)

    """
    _hyperparameters = {'period': [int, (1, np.inf)]}

    def __init__(self, period=2):
        self.period = period

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        check_array(X)

        X = X.reshape((-1, 1))

        self._n_features = X.shape[1]
        return self

    def transform(self, X, y=None):
        """Transform/resample X.

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
        # Check if fit had been called
        check_is_fitted(self, ['_n_features'])
        X = check_array(X, copy=True)

        X = X.reshape((-1, self._n_features))
        Xt = X[::self.period]

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
            ``n_samples_new = n_samples // period``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_n_features'])
        yt = column_or_1d(y).copy()

        yt = y[::self.period]
        return yt


class Stationarizer(BaseEstimator, TransformerResamplerMixin):
    """Data sampling transformer that returns numpy.ndarray.

    Parameters
    ----------
    operation : str, default: 'return'
        The type of stationarization operation with which to stationarize
        the time series. It can have two values:

        - 'return':
            This option transforms the time series :math:`{X_t}_t` into the
            time series of relative returns, i.e. the ratio :math:`(X_t-X_{
            t-1})/X_t`.

        - 'log-return':
            This option transforms the time series :math:`{X_t}_t` into the
            time series of relative log-returns, i.e. :math:`\\log(X_t/X_{
            t-1})`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from giotto.time_series import Stationarizer
    >>> # Create a noisy signal sampled
    >>> signal = np.asarray([np.sin(x /40) + 5 + np.random.random()
    >>> for x in range(0, 300)]).reshape(-1, 1)
    >>> plt.plot(signal)
    >>> plt.show()
    >>> # Initialize the stationarizer
    >>> stationarizer = Stationarizer(stationarization_type='return')
    >>> stationarizer.fit(signal)
    >>> signal_stationarized = stationarizer.transform(signal)
    >>> plt.plot(signal_stationarized)

    """
    _hyperparameters = {'operation': [str, ['return', 'log-return']]}

    def __init__(self, operation='return'):
        self.operation = operation

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        check_array(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transform/resample X.

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
            ``n_samples_new = n_samples - 1``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        Xt = check_array(X, copy=True)

        if self.operation == 'return':
            Xt = np.diff(Xt, n=1, axis=0) / Xt[1:, :]
        else:  # 'log-return' operation
            Xt = np.diff(np.log(Xt), n=1, axis=0)

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
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        yt = column_or_1d(y).copy()

        yt = y[1:]

        return yt
