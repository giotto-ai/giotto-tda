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


def betti(diagram, sampling):
    if diagram.size == 0:
        return np.zeros(sampling.shape)

    born = sampling >= diagram[:, 0]
    notDead = sampling < diagram[:, 1]
    alive = np.logical_and(born, notDead)
    betti = np.sum(alive, axis=1).T
    return betti

def landscape(diagram, n_layers, sampling):
    if diagram.size == 0:
        return np.hstack([np.zeros(sampling.shape)] * n_layers)

    midpoints = (diagram[:, 1] + diagram[:, 0]) * m.sqrt(2) / 2.
    heights = (diagram[:, 1] - diagram[:, 0]) * m.sqrt(2) / 2.

    mountains = [ -np.abs(sampling - midpoints[i]) + heights[i] for i in range(len(diagram)) ]
    fibers = np.vstack([np.where(mountains[i] > 0, mountains[i], 0) for i in range(len(diagram))])

    lastLayer = fibers.shape[0]-1
    landscape = np.flip(np.partition(fibers, range(lastLayer-n_layers, lastLayer, 1), axis=0)[-n_layers:, :], axis=0)
    return landscape

def kernel_landscape_distance(x, y, dimension, sampling=None, order=2, n_layers=1):
    landscapeX = landscape(x, n_layers, sampling[dimension])
    landscapeY = landscape(y, n_layers, sampling[dimension])
    return np.linalg.norm(landscapeX - landscapeY, ord=order)

def kernel_betti_distance(x, y, dimension, sampling=None, order=2):
    bettiX = betti(x, sampling[dimension])
    bettiY = betti(y, sampling[dimension])
    return np.linalg.norm(bettiX - bettiY, ord=order)

def bottleneck_distance(x, y, dimension=None):
    return gudhi_bottleneck_distance(x, y)

def wasserstein_distance(x, y, dimension=None, order=1):
    return hera_wasserstein_distance(x, y, order)

class DiagramDistance(BaseEstimator, TransformerMixin):
    implementedMetricRecipes = {'bottleneck': bottleneck_distance, 'wasserstein': wasserstein_distance,
                                'landscape': kernel_landscape_distance, 'betti': kernel_betti_distance}

    def __init__(self, metric_kwargs={'metric': 'bottleneck'}, order=np.inf, n_jobs=1):
        self.metric_kwargs = metric_kwargs
        self.order = order
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'metric_kwargs': self.metric_kwargs, 'order': self.order, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        pass

    def _parallel_pairwise(self, X, Y, iterator, outputShape, n_jobs):
        metric_kwargs = self.metric_kwargs.copy()

        if 'metric' in metric_kwargs:
            metric_kwargs.pop('metric')

        if 'n_samples' in metric_kwargs:
            metric_kwargs.pop('n_samples')

        distanceList = Parallel(n_jobs = n_jobs) ( delayed(self.metric) (X[dimension][i,:,:], Y[dimension][j,:,:], dimension, **metric_kwargs)
                                                        for i, j in iterator for dimension in X.keys())
        distanceArray = np.array(distanceList).reshape(outputShape)
        return distanceArray

    def _parallel_matrix(self, X, Y, isSame):
        #    X, Y = check_pairwise_arrays(X, Y)
        numberDimensions = len(X)
        numberDiagramsX = next(iter(X.values())).shape[0]

        if isSame:
            distances = np.zeros((numberDiagramsX, numberDiagramsX))
            # Only calculate metric for upper triangle
            iterator = list(itertools.combinations(range(numberDiagramsX), 2))

            distanceArray = self._parallel_pairwise(X, X, iterator, (len(iterator), numberDimensions), n_jobs)
            distanceArray = np.linalg.norm(distanceArray, axis=1, ord=self.order)
            distanceMatrix[tuple(zip(*iterator))] = distanceArray
            distanceMatrix = distanceMatrix + distanceMatrix.T

        else:
            numberDiagramsY = next(iter(Y.values())).shape[0]
            # Calculate all cells
            iterator = tuple(itertools.product(range(numberDiagramsX), range(numberDiagramsY)))

            distanceMatrix = self._parallel_pairwise(X, X, iterator, (numberDiagramsX, numberDiagramsY, numberDimensions), n_jobs)
            distanceMatrix = np.linalg.norm(distanceMatrix, axis=2, ord=self.order)
        return distanceMatrix

    def fit(self, XList, y=None):
        self._validate_params()
        self.isFitted = True

        self.metricName = self.metric_kwargs['metric']
        self.metric = self.implementedMetricRecipes[self.metricName]

        if 'n_samples' in self.metric_kwargs:
            self.n_samples = self.metric_kwargs['n_samples']

        self._X = XList[0]

        # As it will be passed to the metrics, it has to be initialized even if the metric is not using sampling
        self.sampling = { dimension: None for dimension in self._X.keys() }

        if self.metricName in ['landscape', 'betti']:
            maximumPersistences = { dimension: np.max(self._X[dimension][:,:,1]) * m.sqrt(2) if self._X[dimension][:,:,0].size != 0
                                   else -np.inf for dimension in self._X.keys() }
            maximumPersistence = max(list(maximumPersistences.values()))
            maximumPersistences = { dimension: maximumPersistences[dimension] if maximumPersistences[dimension] != -np.inf else maximumPersistence
                                    for dimension in self._X.keys() }

            minimumPersistences = { dimension: np.min(self._X[dimension][:,:,0]) * m.sqrt(2) if self._X[dimension][:,:,0].size != 0
                                   else np.inf for dimension in self._X.keys() }
            minimumPersistence = min(list(minimumPersistences.values()))
            minimumPersistences = { dimension: minimumPersistences[dimension] if minimumPersistences[dimension] != np.inf else minimumPersistence
                                    for dimension in self._X.keys() }

            stepPersistences = { dimension: (maximumPersistences[dimension]-minimumPersistences[dimension])/self.n_samples for dimension in self._X.keys() }
            self.metric_kwargs['sampling'] = { dimension: np.arange(minimumPersistences[dimension], maximumPersistences[dimension],
                                                                        stepPersistences[dimension]).reshape((-1, 1))
                                               for dimension in self._X.keys() }

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
            XTransformed = self._parallel_matrix({ dimension: np.vstack([self._X[dimension], X[dimension]]) for dimension in self._X.keys()}, X, isSame=False)
        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed
