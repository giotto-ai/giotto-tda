import numpy as np
import pandas as pd
import sklearn as sk

from joblib import Parallel, delayed

from pandas.core.resample import Resampler as rsp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
import math as m

class TakensEmbedding(BaseEstimator, TransformerMixin):
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
    inputShape : tuple The shape the data passed to :meth:`fit`
    """

    implementedEmbeddingParametersType = ['fixed', 'search']

    def __init__(self, outerWindowDuration=20, outerWindowStride=2, embeddingParametersType=embeddingDimension,
                 'search'=5, embeddingTimeDelay=1, embeddingStride=1, n_jobs=1):
        self.outerWindowDuration = outerWindowDuration
        self.outerWindowStride = outerWindowStride
        self.embeddingParametersType = embeddingParametersType
        self.embeddingDimension = embeddingDimension
        self.embeddingTimeDelay = embeddingTimeDelay
        self.embeddingStride = embeddingStride
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'outerWindowDuration': self.outerWindowDuration,
                'outerWindowStride': self.outerWindowStride,
                'embeddingParametersType': self.embeddingParametersType,
                'embeddingDimension': self.embeddingDimension,
                'embeddingTimeDelay': self.embeddingTimeDelay,
                'embeddingStride': self.embeddingStride,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(embeddingParametersType):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        if embeddingParametersType not in Sampling.implementedEmbeddingParametersTypes:
            raise ValueError('The embedding parameters type you specified is not implemented')

    @staticmethod
    def _embed(X, outerWindowDuration, outerWindowStride, embeddingTimeDelay, embeddingDimension, embeddingStride=1):
        numberOuterWindows = (X.shape[0] - outerWindowDuration) // outerWindowStride + 1
        numberInnerWindows = (outerWindowDuration - embeddingTimeDelay*embeddingDimension) // embeddingStride + 1

        X = np.flip(X)

        XEmbedded = np.stack( [
            np.stack( [
                X[i*outerWindowStride + j*embeddingStride
                  : i*outerWindowStride + j*embeddingStride + embeddingTimeDelay*embeddingDimension
                  : embeddingTimeDelay].flatten()
                for j in range(0, numberInnerWindows) ] )
            for i in range(0, numberOuterWindows) ])

        return np.flip(XEmbedded).reshape((1, numberInnerWindows, embeddingDimension))

    @staticmethod
    def _mutual_information(X, embeddingTimeDelay, numberBins):
        """This function calculates the mutual information given the delay
        """
        contingency = np.histogram2d(X[:-embeddingTimeDelay], X[embeddingTimeDelay:], bins=numberBins)[0]
        mutualInformation = mutual_info_score(None, None, contingency=contingency)
        return mutualInformation

    @staticmethod
    def _false_nearest_neighbors(X, embeddingTimeDelay, embeddingDimension, embeddingStride=1):
        """Calculates the number of false nearest neighbours of embedding dimension"""
        XEmbedded = TakensEmbedding._embed(X, X.shape[0], 1, embeddingTimeDelay, embeddingDimension, embeddingStride)
        XEmbedded = XEmbedded.reshape((XEmbedded.shape[1], XEmbedded.shape[2]))

        neighbor = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(XEmbedded)
        distances, indices = neighbor.kneighbors(XEmbedded)
        distance = distances[:, 1]
        XNeighbor = X[indices[:, 1]]

        epsilon = 2.0 * np.std(X)
        tolerance = 10

        nonZeroDistance = distance[:-embeddingDimension*embeddingTimeDelay] > 0
        falseNeighborCriteria = np.abs(np.roll(X, -embeddingDimension*embeddingTimeDelay)[X.shape[0]-XEmbedded.shape[0]:-embeddingDimension*embeddingTimeDelay]
                                       - np.roll(XNeighbor, -embeddingDimension*embeddingTimeDelay)[:-embeddingDimension*embeddingTimeDelay]) \
                                       / distance[:-embeddingDimension*embeddingTimeDelay] > tolerance
        limitedDatasetCriteria = distance[:-embeddingDimension*embeddingTimeDelay] < epsilon
        numberFalseNeighbors = np.sum(nonZeroDistance * falseNeighborCriteria * limitedDatasetCriteria)
        return numberFalseNeighbors

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
        self._validate_params(self.embeddingParametersType)

        if type(XList) is list:
            X = XList[0]
        else:
            X = XList

        if self.embeddingParametersType == 'search':
            mutualInformationList = Parallel(n_jobs=self.n_jobs) ( delayed(self._mutual_information) (X, embeddingTimeDelay, numberBins=100)
                                                               for embeddingTimeDelay in range(1, self.embeddingTimeDelay+1) )
            self.embeddingTimeDelay = mutualInformationList.index(min(mutualInformationList)) + 1

            numberFalseNeighborsList = Parallel(n_jobs=self.n_jobs) ( delayed(self._false_nearest_neighbors) (X, self.embeddingTimeDelay, embeddingDimension, embeddingStride=1)
                                                                 for embeddingDimension in range(1, self.embeddingDimension+3) )
            variationList = [ np.abs(numberFalseNeighborsList[embeddingDimension-1]-2*numberFalseNeighborsList[embeddingDimension]+numberFalseNeighborsList[embeddingDimension+1]) \
                              /(numberFalseNeighborsList[embeddingDimension] + 1)/embeddingDimension for embeddingDimension in range(1, self.embeddingDimension+1) ]
            self.embeddingDimension = variationList.index(min(variationList)) + 1

        self.isFitted = True
        return self

    def transform(self, XList, y=None):
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
            X = XList[0]
        else:
            X = XList

        if XData.shape[0] < self.outerWindowDuration:
            raise ValueError('Not enough data to have a single outer window.')

        XTransformed = self._embed(X, self.outerWindowDuration, self.outerWindowStride, self.embeddingTimeDelay, self.embeddingDimension, self.embeddingStride)
        XListTransformed.append(XTransformed)

        XTransformed = self._embed(X, self.outerWindowDuration, self.outerWindowStride, embeddingTimeDelay=1, embeddingDimension=self.outerWindowDuration, embeddingStride=0)
        XListTransformed.append(XTransformed)

        if type(XList) is list:
            if len(XList) >= 2:
                y = XList[1]
                yTransformed =  self._embed(y, self.outerWindowDuration, self.outerWindowStride, embeddingTimeDelay=1, embeddingDimension=self.outerWindowDuration, embeddingStride=0)
                XListTransformed.append(yTransformed)

        return XListTransformed
