import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Resampler(BaseEstimator, TransformerMixin):
    """Data sampling transformer that returns a sampled numpy.ndarray.

    Parameters
    ----------
    period : int, default: 2
        The sampling period for periodic sampling.


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
    def __init__(self, period=2):
        self.period = period

    def _validate_params(self):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:``fit`` are valid.
        """
        if not isinstance(self.period, int):
            raise TypeError('The period ',  self.period, ' is not an int')

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.

        """
        self._validate_params()

        X = X.reshape((-1, 1))

        self._n_features = X.shape[1]
        return self

    def transform(self, X, y=None):
        """Resample X according to the 'sampling_type'.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples_new, n_features)
            The resampled array. ``n_samples_new`` depends on
            ``sampling_period`` if ``sampling_type`` is 'periodic' or on the
            number of ``sampling_times`` if ``sampling_type`` is 'fixed'.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_n_features'])

        X = X.reshape((-1, 1))
        X_transformed = X[::self.period]

        return X_transformed


class Stationarizer(BaseEstimator, TransformerMixin):
    """Data sampling transformer that returns numpy.ndarray.

    Parameters
    ----------
    stationarization_type : str, default: 'return'
        The type of stationarization technique with which to stationarize
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
    valid_stationarization_types = ['return', 'log-return']

    def __init__(self, stationarization_type='return'):
        self.stationarization_type = stationarization_type

    @staticmethod
    def _validate_params(stationarization_type):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.

        """
        if stationarization_type not in \
                Stationarizer.valid_stationarization_types:
            raise ValueError(
                'The transformation type %s is not supported' %
                stationarization_type)

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
            Returns self.

        """
        self._validate_params(self.stationarization_type)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Stationarize X according to ``stationarization_type``.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples - 1, n_features)
            The array containing the stationarized inputs.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = X
        if self.stationarization_type == 'return':
            X_transformed = np.diff(X_transformed, n=1,
                                    axis=0) / X_transformed[1:, :]
        else:  # 'log-return' stationarization type
            X_transformed = np.log(X_transformed)
            X_transformed = np.diff(X_transformed, n=1, axis=0)

        return X_transformed
