import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as rsp
import datetime as dt

class Resampler(BaseEstimator, TransformerMixin):
    """
    data sampling transformer that returns a sampled Pandas dataframe with a datetime index

    Parameters
    ----------
    samplingType : str
        The type of sampling

    samplingPeriod : str
        Time anchors giving the period of the sampling. Used only if samplingType is 'periodic'

    samplingTimeList : list of datetime
        Datetime at which the samples should be taken. Used only if samplingType is 'fixed'

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
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
           of the :meth:'fit' are valid.
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
    data sampling transformer that returns a sampled Pandas dataframe with a datetime index

    Parameters
    ----------
    samplingType : str
        The type of sampling

    samplingPeriod : str
        Time anchors giving the period of the sampling. Used only if samplingType is 'periodic'

    samplingTimeList : list of datetime
        Datetime at which the samples should be taken. Used only if samplingType is 'fixed'

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
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
