import numpy as np
import pandas as pd
import sklearn as sk
from pandas.core.resample import Resampler as rsp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

class TakensEmbedder(BaseEstimator, TransformerMixin):
    """
    Transformer that return a time serie embedded according to Taken's sliding window.

    Parameters
    ----------
    nPeriodsList : list of int
        List of number of periods

    outerWindowDuration: pd.time
        Duration of the outer sliding window

    outerWindowStride: pd.time
        Stride of the outer sliding window

    innerWindowDuration: pd.time
        Duration of the inner sliding window

    innerWindowStride: pd.time
        Stride of the inner sliding window

    Attributes
    ----------
    inputShape : tuple
        The shape the data passed to :meth:`fit`
    """

    def __init__(self, outerWindowDuration = 20, outerWindowStride = 2,
                 innerWindowDuration = 5, innerWindowStride = 1):
        self.outerWindowDuration = outerWindowDuration
        self.outerWindowStride = outerWindowStride
        self.innerWindowDuration = innerWindowDuration
        self.innerWindowStride = innerWindowStride

    def get_params(self, deep=True):
        return {'outerWindowDuration': self.outerWindowDuration,
                'outerWindowStride': self.outerWindowStride,
                'innerWindowDuration': self.innerWindowDuration,
                'innerWindowStride': self.innerWindowStride}

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         self.setattr(parameter, value)
    #     return self

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """

    def fit(self, XList, y=None):
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
        self._validate_params()

        self.isFitted = True
        return self

    def transform(self, XList, y = None):
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

        XListTransformed = []
        if type(XList) is list:
            XData = XList[0]
        else:
            XData = XList

        if XData.shape[0] < self.outerWindowDuration:
            raise ValueError('Not enough data to have a single outer window.')

        numberOuterWindows = (XData.shape[0] - self.outerWindowDuration) // self.outerWindowStride + 1
        numberInnerWindows = (self.outerWindowDuration - self.innerWindowDuration) // self.innerWindowStride + 1

        XTransformed = np.stack( [
            np.stack( [
                XData.values[i*self.outerWindowStride + j*self.innerWindowStride
                             :i*self.outerWindowStride + j*self.innerWindowStride + self.innerWindowDuration].flatten()
                for j in range(0, numberInnerWindows) ] )
            for i in range(0, numberOuterWindows) ])

        XListTransformed.append(XTransformed)

        numberInnerWindows = 1

        if type(XList) is list:
            XData = XList[0]
        else:
            XLabel = XList

        XTransformed = np.stack( [
            np.stack( [
                XLabel.values[i*self.outerWindowStride + j
                              :i*self.outerWindowStride + j + self.outerWindowDuration].flatten()
                for j in range(0, numberInnerWindows) ] )
            for i in range(0, numberOuterWindows) ])

        XListTransformed.append(XTransformed.reshape((XTransformed.shape[0], -1)))

        if type(XList) is list:
            if len(XList) >= 2:
                XLabel = XList[1]
                XTransformed = np.stack( [
                    np.stack( [
                        XLabel.values[i*self.outerWindowStride + j
                                      :i*self.outerWindowStride + j + self.outerWindowDuration].flatten()
                        for j in range(0, numberInnerWindows) ] )
                    for i in range(0, numberOuterWindows) ])

                XListTransformed.append(XTransformed)

        return XListTransformed
