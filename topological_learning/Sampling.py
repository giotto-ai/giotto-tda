import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as rsp
import datetime as dt

class Sampling(BaseEstimator, TransformerMixin):
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

    implementedTransformationTypes = ['none', 'return', 'log-return']
    implementedSamplingTypes = ['periodic', 'fixed']

    def __init__(self, transformationType = 'none', samplingType = 'periodic', samplingPeriod = '2h',
                 samplingTimeList=[dt.time(0,0,0)], removeWeekends = True):
        self.transformationType = transformationType
        self.samplingType = samplingType
        self.samplingPeriod = samplingPeriod
        self.samplingTimeList = samplingTimeList
        self.removeWeekends = removeWeekends

    def get_params(self, deep=True):
        return {'samplingType': self.samplingType,
                'samplingPeriod': self.samplingPeriod,
                'removeWeekends': self.removeWeekends}

    @staticmethod
    def _validate_params(transformationType, samplingType):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        if transformationType not in Sampling.implementedTransformationTypes:
            raise ValueError('The transformation type you specified is not implemented')

        if samplingType not in Sampling.implementedSamplingTypes:
            raise ValueError('The sampling type you specified is not implemented')

    def fit(self, XArray, y = None):
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
        self._validate_params(self.transformationType, self.samplingType)

        self.isFitted = True
        return self

    def transform(self, XArray, y=None):
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
        check_is_fitted(self, ['isFitted'])

        if self.samplingType == 'periodic':
            XArray = XArray.resample(self.samplingPeriod).first()
        elif self.samplingType == 'fixed':
            XArray = XArray.iloc[np.isin(XArray.index.time, self.samplingTimeList)].copy()

        if 'return' in self.transformationType:
            XArray.iloc[:, 0] = XArray.iloc[:, 0].pct_change(1)
            XArray.drop(XArray.index[0], inplace=True)

        if 'log' in self.transformationType:
            XArray.iloc[:, 0] = np.log(XArray.iloc[:, 0])

        if self.removeWeekends:
            XArray = XArray[:][XArray.index.dayofweek < 5]

        XArray.columns = range(len(XArray.columns))
        if len(XArray.columns) == 1:
            return [ XArray.iloc[:, 0].values ]
        else:
            return [ XArray.iloc[:, 0].values, XArray.iloc[:, 1:].values ]
