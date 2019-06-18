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

from .DiagramDistance import DiagramDistance

class DiagramFiltering(BaseEstimator, TransformerMixin):

    def __init__(self, delta=0., diagramDistance=DiagramDistance(), epsilon=1., tolerance=1e-2, n_jobs=1):
        self.delta = delta
        self.diagramDistance = diagramDistance
        self.epsilon = epsilon
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'delta': self.delta}

    # {'diagramDistance': self.diagramDistance, 'epsilon': self.epsilon, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        pass

    @staticmethod
    def _bisection(X, diagramDistance, tolerance):
        iterator = iter([(i, i) for i in range(len(X))])

        lowerCutoff = 0.
        upperCutoff = 1.

        delta = 1.

        while distance > tolerance:
            middleCutoff = (lowerCutoff + upperCutoff) / 2.
            XFiltered = self._filter(X, middleCutoff)
            distance = diagramDistance._parallel_pairwise(X, XFiltered, iterator, self.n_jobs)

            if distance == tolerance:
                return middleCutoff
            elif (distance - tolerance)*() < 0:
                upperCutoff = middleCutoff
            else:
                lowerCutoff = middleCutoff

        return middleCutoff

    @staticmethod
    def _filter(XScaled, cutoff):
        XFiltered = { dimension: X.copy() for dimension, X in XScaled.items() }
        mask = { dimension: m.sqrt(2)/2. * (X[:, :, 1] - X[:, :, 0]) <= cutoff for dimension, X in XFiltered.items() }

        for dimension, X in XFiltered.items():
            X[mask[dimension], :] = [0, 0]

        maxPoints = { dimension: np.max(np.sum(X[:, :, 1] != 0, axis=1)) for dimension, X in XFiltered.items() }
        XFiltered = { dimension: X[:, :maxPoints[dimension], :] for dimension, X in XFiltered.items() }
        return XFiltered

    def fit(self, XList, y=None):
        self._validate_params()

        X = XList[0]

        self.isFitted = True
        self.diagramDistance.fit(XList)

        # self.delta = self.delta

        return self

    def transform(self, XList, y=None):
        # Check is fit had been called
        #check_is_fitted(self, ['isFitted'])
        XListTransformed = []

        XTransformed = self._filter(XList[0], self.delta)
        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed
