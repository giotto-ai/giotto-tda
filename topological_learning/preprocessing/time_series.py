import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as rsp
import datetime as dt

class Resampler(BaseEstimator, TransformerMixin):
    """
    Data sampling transformer that returns a sampled Pandas dataframe with a datetime index.

    Parameters
    ----------
    sampling_type : str, optional, default: 'periodic'
        The type of sampling. Its value can be either 'periodic' or 'fixed':

        - 'periodic':
            It means sampling with a constant ``sampling_period``.
        - 'fixed':
            It entails that the list of sampling times has to be provided via the parameter
            ``sampling_times``.

    sampling_period : str, optional, default: '2h'
        The sampling period for periodic sampling. Used only if ``sampling_type`` is 'periodic'.

    sampling_times : list of datetime, optional, default: [dt.time(0,0,0)]
        dt.Datetime at which the samples should be taken. Used only if ``sampling_type`` is 'fixed'.

    remove_weekends : boolean, optional, default: True
        Option to remove week-ends from the time series pd.DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import topological_learning.preprocessing as prep
    >>> import matplotlib.pyplot as plt
    >>> # Create a noisy signal sampled
    >>> signal_noise = np.asarray([np.sin(x /40) - 0.5 + np.random.random()
    ... for x in range(0,300)])
    >>> # Set up the dataframe of the z-projection of the simulated solution
    >>> df_noise = pd.DataFrame(signal_noise)
    >>> index = pd.to_datetime(df_noise.index, utc=True, unit='h')
    >>> df_noise.index = index
    >>> # Set up the Sampler
    >>> sampling_period = '10h'
    >>> periodic_sampler = prep.Resampler(sampling_type='periodic', sampling_period=sampling_period,
    ...                                   remove_weekends=False)
    >>> # Fit and transform the DataFrame
    >>> periodic_sampler.fit(df_noise)
    >>> df_noise_sampled = periodic_sampler.transform(df_noise)
    >>> plt.plot(df_noise_sampled)
    >>> plt.plot(df_noise)


    """
    implemented_sampling_types = ['periodic', 'fixed']

    def __init__(self, sampling_type = 'periodic', sampling_period = '2h',
                 sampling_times=[dt.time(0,0,0)], remove_weekends = True):
        self.sampling_type = sampling_type
        self.sampling_period = sampling_period
        self.sampling_times = sampling_times
        self.remove_weekends = remove_weekends

    def get_params(self, deep=True):
        return {'sampling_type': self.sampling_type,
                'sampling_period': self.sampling_period,
                'sampling_times': self.sampling_times,
                'remove_weekends': self.remove_weekends}

    @staticmethod
    def _validate_params(sampling_type):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:``fit`` are valid.
        """
        if sampling_type not in Resampler.implemented_sampling_types:
            raise ValueError('The sampling type you specified is not implemented')

    def fit(self, X, y = None):
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
        self._validate_params(self.sampling_type)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Resample X according to the 'sampling_type'.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples_new, n_features)
            The resampled array. ``n_samples_new`` depends on ``sampling_period`` if ``sampling_type``
            is 'periodic' or on the number of ``sampling_times`` if ``sampling_type`` is 'fixed'.

        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        if self.sampling_type == 'periodic':
            X = X.resample(self.sampling_period).first()
        elif self.sampling_type == 'fixed':
            X = X.iloc[np.isin(X.index.time, self.sampling_times)].copy()

        if self.remove_weekends:
            X = X[:][X.index.dayofweek < 5]

        X.columns = range(len(X.columns))
        X_transformed = X.iloc[:, 0].values
        return X_transformed


class Stationarizer(BaseEstimator, TransformerMixin):
    """
    Data sampling transformer that returns a stationarized Pandas dataframe with a datetime index.

    Parameters
    ----------
    sationarization_type : str, default: 'return'
        The type of stationarization technique with which to stationarize the time series. It can
        have two values:

        - 'return':
            This option transforms the time series :math:`{X_t}_t` into the time series of relative
            returns, i.e. the ratio :math:`(X_t-X_{t-1})/X_t`.

        - 'log-return':
            This option transforms the time series :math:`{X_t}_t` into the time series of relative
            log-returns, i.e. :math:`\\log(X_t/X_{t-1})`.

    Examples
    --------
    >>> import topological_learning.preprocessing as prep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Create a noisy signal sampled
    >>> signal_noise = np.asarray([np.sin(x /40) - 0.5 + np.random.random() for x in range(0,300)])
    >>> # Initzialize the stationarizer
    >>> return_stationarizer = prep.Stationarizer(stationarization_type='return')
    >>> return_stationarizer.fit(signal_noise)
    >>> signal_noise_stationarized = return_stationarizer.transform(signal_noise)
    >>> plt.plot(signal_noise_stationarized)

    """
    implemented_stationarization_types = ['return', 'log-return']

    def __init__(self, stationarization_type='return'):
        self.stationarization_type = stationarization_type

    def get_params(self, deep=True):
        return {'stationarization_type': self.stationarization_type}

    @staticmethod
    def _validate_params(stationarization_type):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        if stationarization_type not in Stationarizer.implemented_stationarization_types:
            raise ValueError('The transformation type you specified is not implemented')

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
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = X
        if 'return' in self.stationarization_type:
            X_transformed = np.diff(X_transformed, n=1, axis=0)/ X_transformed[1:, :]

        if 'log' in self.stationarization_type:
            X_transformed = np.log(X_transformed)

        print(X_transformed.shape)

        return X_transformed
