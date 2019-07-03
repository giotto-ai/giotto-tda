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
        The sampling period for periodic sampling. Used only if samplingType is 'periodic'.
        
    sampling_times : list of datetime, optional, default: ``dt.time(0,0,0)``
        Datetime at which the samples should be taken. Used only if samplingType is 'fixed'.
        
    remove_weekends : boolean, optional, default: True
        Option to remove week-ends from the time-series DataFrame.

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import topological_learning.preprocessing as prep
    >>> import matplotlib.pyplot as plt
    >>> # Create a noisy signal sampled
    >>> signal_noise = np.asarray([np.sin(x /40) - 0.5 + np.random.random() for x in range(0,300)])
    >>> # Set up the dataframe of the z-projection of the simulated solution
    >>> zDataFrame = pd.DataFrame(signal_noise)
    >>> index = pd.to_datetime(zDataFrame.index, utc=True, unit='h')
    >>> zDataFrame.index = index
    >>> # Set up the Sampler
    >>> samplingPeriod = '10h'
    >>> periodicSampler = prep.Resampler(sampling_type='periodic', sampling_period=samplingPeriod,
    ... remove_weekends=False)
    >>> # Fit and transform the DataFrame
    >>> periodicSampler.fit(zDataFrame)
    >>> zDataFrameSampled = periodicSampler.transform(zDataFrame)
    >>> plt.plot(zDataFrameSampled)
    >>> plt.plot(zDataFrame)
    
    
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
        self._validate_params(self.sampling_type)

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
    Data sampling transformer that returns a a stationarized Pandas dataframe with a datetime index

    Parameters
    ----------
    sationarization_type : str, default: 'none'
        The type of stationarization technique with whcih to stationarize the time-series. It can
        have three values:
        
        - 'none':
            No stationarization applied to the time-series.
            
        - 'return':
            This option transforms the time series {X_t}_t into the time-series of relative
            returns, i.e. the ratio (X_t-X_{t-1})/X_t * 100.
            
        - 'log-return':
            This option transforms the time-series series {X_t}_t into the time-series of relative
            log-returns, i.e. log(X_t/X_{t-1}).

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
        
    Examples
    --------
    >>> import topological_learning.preprocessing as prep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Create a noisy signal sampled
    >>> signal_noise = np.asarray([np.sin(x /40) - 0.5 + np.random.random() for x in range(0,300)])
    >>> # Initzialize the stationarizer
    >>> returnStationarizer = prep.Stationarizer(stationarization_type='return')
    >>> returnStationarizer.fit(signal_noise)
    >>> zDataFrameStationarized = returnStationarizer.transform(signal_noise)
    >>> plt.plot(zDataFrameStationarized)
    """

    implemented_stationarization_types = ['none', 'return', 'log-return']

    def __init__(self, stationarization_type = 'none'):
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
        self._validate_params(self.stationarization_type)

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

        X_transformed = X
        if 'return' in self.stationarization_type:
            X_transformed =np.diff(X_transformed)/ X_transformed[1:] * 100.

        if 'log' in self.stationarization_type:
            X_transformed = np.log(X_transformed)

        return X_transformed
