# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
#          Matteo Caorsi <m.caorsi@l2f.ch>
# License: TBD

import numpy as np
import math as m
from sklearn.utils._joblib import Parallel, delayed

from gudhi import bottleneck_distance as gudhi_bottleneck_distance
from ..external.bindings import wasserstein as hera_wasserstein_distance


def betti_function(diagram, sampling):
    if diagram.size == 0:
        return np.zeros(sampling.shape)

    born = sampling >= diagram[:, 0]
    not_dead = sampling < diagram[:, 1]
    alive = np.logical_and(born, not_dead)
    betti = np.sum(alive, axis=1).T
    return betti


def landscape_function(diagram, n_layers, sampling):
    if diagram.size == 0:
        return np.hstack([np.zeros(sampling.shape)] * n_layers)

    midpoints = (diagram[:, 1] + diagram[:, 0]) * m.sqrt(2) / 2.
    heights = (diagram[:, 1] - diagram[:, 0]) * m.sqrt(2) / 2.

    mountains = [-np.abs(sampling - midpoints[i]) +
                 heights[i] for i in range(len(diagram))]
    fibers = np.vstack([np.where(mountains[i] > 0,
                                 mountains[i],
                                 0) for i in range(len(diagram))])

    last_layer = fibers.shape[0] - 1
    landscape = np.flip(np.partition(
        fibers,
        range(last_layer - n_layers, last_layer, 1),
        axis=0)[-n_layers:, :], axis=0)
    return landscape


def kernel_landscape_distance(diagram_x, diagram_y, dimension, sampling=None,
                              order=2, n_layers=1):
    landscape_x = landscape_function(diagram_x, n_layers, sampling[dimension])
    landscape_y = landscape_function(diagram_y, n_layers, sampling[dimension])
    return np.linalg.norm(landscape_x - landscape_y, ord=order)


def kernel_betti_distance(diagram_x, diagram_y, dimension, sampling=None,
                          order=2):
    betti_x = betti_function(diagram_x, sampling[dimension])
    betti_y = betti_function(diagram_y, sampling[dimension])
    return np.linalg.norm(betti_x - betti_y, ord=order)


def bottleneck_distance(diagram_x, diagram_y, dimension=None, order=np.inf,
                        delta=0.0):
    return gudhi_bottleneck_distance(diagram_x, diagram_y, delta)

def wasserstein_distance(diagram_x, diagram_y, dimension=None, order=1,
                         delta=0.0):
    return hera_wasserstein_distance(diagram_x, diagram_y, order, delta)


implemented_metric_recipes = {'bottleneck': bottleneck_distance,
                              'wasserstein': wasserstein_distance,
                              'landscape': kernel_landscape_distance,
                              'betti': kernel_betti_distance}


def _parallel_pairwise(X, Y, metric, metric_params, iterator, n_jobs):
    n_diagrams_X = list(X.values())[0].shape[0]
    n_diagrams_Y = list(Y.values())[0].shape[0]
    n_dimensions = len(X.keys())
    metric_func = implemented_metric_recipes[metric]

    distance_matrix = np.zeros((n_diagrams_X, n_diagrams_Y))
    distance_array = Parallel(n_jobs=n_jobs)(delayed(metric_func)(
        X[dimension][i, :, :], Y[dimension][j, :, :],
        dimension, **metric_params)
        for i, j in iterator for dimension in X.keys())
    distance_array = np.array(distance_array). \
        reshape((len(iterator), n_dimensions))
    distance_array = np.linalg.norm(distance_array, axis=1,
                                    ord=metric_params['order'])
    distance_matrix[tuple(zip(*iterator))] = distance_array
    return distance_matrix


def kernel_landscape_norm(diagram, dimension, sampling=None,
                          order=2, n_layers=1):
    landscape = landscape_function(diagram, n_layers, sampling[dimension])
    return np.linalg.norm(landscape, ord=order)


def kernel_betti_norm(diagram, dimension, sampling=None, order=2):
    betti = betti_function(diagram, sampling[dimension])
    return np.linalg.norm(betti, ord=order)


def bottleneck_norm(diagram, dimension=None, order=np.inf):
    return np.linalg.norm(m.sqrt(2) / 2. * (diagram[:, 1] - diagram[:, 0]),
                          ord=order)


def wasserstein_norm(diagram, dimension=None, order=1):
    return np.linalg.norm(m.sqrt(2) / 2. * (diagram[:, 1] - diagram[:, 0]),
                          ord=order)


implemented_norm_recipes = {'bottleneck': bottleneck_norm,
                            'wasserstein': wasserstein_norm,
                            'landscape': kernel_landscape_norm,
                            'betti': kernel_betti_norm}


def _parallel_norm(X, metric, metric_params, n_jobs):
    n_dimensions = len(X.keys())
    norm_func = implemented_norm_recipes[metric]

    norm_array = Parallel(n_jobs=n_jobs)(delayed(norm_func)(
        X[dimension][i, :, :], dimension, **metric_params)
        for i in range(next(iter(X.values())).shape[0])
        for dimension in X.keys())
    norm_array = np.linalg.norm(np.array(norm_array).
                                reshape((-1, n_dimensions)),
                                ord=metric_params['order'], axis=1)
    return norm_array
