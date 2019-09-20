# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
#          Matteo Caorsi <m.caorsi@l2f.ch>
# License: TBD

import numpy as np
import math as m
from sklearn.utils._joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
import numbers

from giotto_wasserstein import wasserstein_distance \
    as pairwise_wasserstein_distance
from giotto_bottleneck import bottleneck_distance \
    as pairwise_bottleneck_distance

available_metric_params = ['order', 'delta', 'n_samples', 'n_layers', 'sigma']

available_metrics = {'bottleneck': [('delta', numbers.Number, (0., 1.))],
                     'wasserstein': [('order', int, (1, np.inf)),
                                     ('delta', numbers.Number, (0., 1.))],
                     'betti': [('order', int, (1, np.inf)),
                               ('n_samples', int, (1, np.inf))],
                     'landscape': [('order', int, (1, np.inf)),
                                   ('n_samples', int, (1, np.inf)),
                                ('n_layers', int, (1, np.inf))],
                     'heat': [('order', int, (1, np.inf)),
                              ('n_samples', int, (1, np.inf)),
                              ('sigma', numbers.Number, (0., np.inf))]}


def betti_function(diagram, sampling):
    born = sampling >= diagram[:, 0]
    not_dead = sampling < diagram[:, 1]
    alive = np.logical_and(born, not_dead)
    betti = np.sum(alive, axis=1).T
    return betti


def landscape_function(diagram, n_layers, sampling):
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


def heat_function(diagram, sigma, sampling):
    heat = np.zeros((sampling.shape[0], sampling.shape[0]))

    sample_step = sampling[1] - sampling[0]

    sampled_diagram = np.array(diagram // sample_step, dtype=int)
    for sampled_point in sampled_diagram[sampled_diagram[:, 1] != 0]:
        heat[sampled_point[0], sampled_point[1]] += 1
        heat[sampled_point[1], sampled_point[0]] -= 1

    heat = gaussian_filter(heat, sigma, mode='reflect')
    return heat


def kernel_landscape_distance(diagram_x, diagram_y, dimension, sampling=None,
                              order=2, n_layers=100, **kw_args):
    landscape_x = landscape_function(diagram_x, n_layers, sampling[dimension])
    landscape_y = landscape_function(diagram_y, n_layers, sampling[dimension])
    return np.linalg.norm(landscape_x - landscape_y, ord=order)


def kernel_betti_distance(diagram_x, diagram_y, dimension, sampling=None,
                          order=2, **kw_args):
    betti_x = betti_function(diagram_x, sampling[dimension])
    betti_y = betti_function(diagram_y, sampling[dimension])
    return np.linalg.norm(betti_x - betti_y, ord=order)


def kernel_heat_distance(diagram_x, diagram_y, dimension, sigma=1.,
                         sampling=None, order=2, **kw_args):
    heat_x = heat_function(diagram_x, sigma, sampling[dimension])
    heat_y = heat_function(diagram_y, sigma, sampling[dimension])
    return np.linalg.norm(heat_x - heat_y, ord=order)


def bottleneck_distance(diagram_x, diagram_y, dimension=None,
                        delta=0.0, **kw_args):
    return pairwise_bottleneck_distance(diagram_x[diagram_x[:, 1] != 0],
                                        diagram_y[diagram_y[:, 1] != 0], delta)


def wasserstein_distance(diagram_x, diagram_y, dimension=None, order=1,
                         delta=0.0, **kw_args):
    return pairwise_wasserstein_distance(diagram_x, diagram_y, order, delta)


implemented_metric_recipes = {'bottleneck': bottleneck_distance,
                              'wasserstein': wasserstein_distance,
                              'landscape': kernel_landscape_distance,
                              'betti': kernel_betti_distance,
                              'heat': kernel_heat_distance}


def _parallel_pairwise(X, Y, metric, metric_params, iterator, order, n_jobs):
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
    distance_array = np.linalg.norm(distance_array, axis=1, ord=order)
    distance_matrix[tuple(zip(*iterator))] = distance_array
    return distance_matrix


def kernel_landscape_amplitude(diagram, dimension, sampling=None,
                          order=2, n_layers=1, **kw_args):
    landscape = landscape_function(diagram, n_layers, sampling[dimension])
    return np.linalg.norm(landscape, ord=order)


def kernel_betti_amplitude(diagram, dimension, sampling=None, order=2,
                           **kw_args):
    betti = betti_function(diagram, sampling[dimension])
    return np.linalg.norm(betti, ord=order)


def kernel_heat_amplitude(diagram, dimension, sampling=None, sigma=1.,
                     order=2, n_layers=1, **kw_args):
    heat = heat_function(diagram, sigma, sampling[dimension])
    return np.linalg.norm(heat, ord=order)


def bottleneck_amplitude(diagram, dimension=None, order=np.inf, **kw_args):
    return np.linalg.norm(m.sqrt(2) / 2. * (diagram[:, 1] - diagram[:, 0]),
                          ord=order)


def wasserstein_amplitude(diagram, dimension=None, order=1, **kw_args):
    return np.linalg.norm(m.sqrt(2) / 2. * (diagram[:, 1] - diagram[:, 0]),
                          ord=order)


implemented_amplitude_recipes = {'bottleneck': bottleneck_amplitude,
                            'wasserstein': wasserstein_amplitude,
                            'landscape': kernel_landscape_amplitude,
                            'betti': kernel_betti_amplitude,
                            'heat': kernel_heat_amplitude}


def _parallel_amplitude(X, metric, metric_params, n_jobs):
    n_dimensions = len(X.keys())
    amplitude_func = implemented_amplitude_recipes[metric]

    amplitude_array = Parallel(n_jobs=n_jobs)(delayed(amplitude_func)(
        X[dimension][i, :, :], dimension, **metric_params)
        for i in range(next(iter(X.values())).shape[0])
        for dimension in X.keys())
    amplitude_array = np.array(amplitude_array).reshape((-1, n_dimensions))
    return amplitude_array
