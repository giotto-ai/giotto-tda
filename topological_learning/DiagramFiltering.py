import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from sklearn.utils._joblib import Parallel, delayed
import itertools

import math as m
import numpy as np

from .dependencies import bottleneck_distance as gudhi_bottleneck_distance
from .dependencies import wasserstein as hera_wasserstein_distance

class DiagramFiltering(BaseEstimator, TransformerMixin):

    def __init__(self, diagramDistance=DiagramDistance(), tolerance=1., n_jobs=1):
        self.diagramDistance = diagramDistance
        self.tolerance = tolerance
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'diagramDistance': self.diagramDistance, 'tolerance': self.tolerance,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        pass

    def _bisection():
        pass

    @staticmethod
    def _filter(X, cutoff):
        mask = m.sqrt(2)/2. * (X[:,:,1] - X[:,:,0]) <= cutoff
        XFiltered = X.copy()
        XFiltered[mask, :] = [0, 0]
        maxPoints = np.max(np.sum(XFiltered[:, :, 1] != 0, axis=1))
        XFiltered = np.sort(XFiltered, axis=1)
        return XFiltered[:, -maxPoints:, :]

    def fit(self, XList, y=None):
        self._validate_params()

        self.isFitted = True
        diagramDistance.fit(XList)

        iterator = iter([(i, i) for i in range(len(XList[0]))])


        return self

    def transform(self, XList, y=None):
        # Check is fit had been called
        #check_is_fitted(self, ['isFitted'])
        X = XList[0]

        XListTransformed = [ ]

        if np.sum([ np.array_equal(X[dimension], self._X[dimension]) for dimension in X.keys()]) == len(X):
            XTransformed = self._parallel_pairwise(X, X, isSame=True)
        else:
            maxBettiNumbers = { dimension: max(self._X[dimension].shape[1], X[dimension].shape[1]) for dimension in self._X.keys()}
            self._X = { dimension: np.pad(self._X[dimension], ((0, 0), (0, maxBettiNumbers[dimension] - self._X[dimension].shape[1]), (0, 0)), 'constant')
                        for dimension in self._X.keys() }
            X = { dimension: np.pad(X[dimension], ((0, 0), (0, maxBettiNumbers[dimension] - X[dimension].shape[1]), (0, 0)), 'constant')
                  for dimension in self._X.keys() }
            XTransformed = self._parallel_pairwise({ dimension: np.vstack([self._X[dimension], X[dimension]]) for dimension in self._X.keys()}, X, isSame=False)
        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed
