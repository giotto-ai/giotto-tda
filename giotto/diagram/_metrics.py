# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
#          Matteo Caorsi <m.caorsi@l2f.ch>
# License: TBD

import numpy as np
from scipy.spatial.distance import cdist, pdist
from giotto_bottleneck import bottleneck_distance \
    as pairwise_bottleneck_distance
from giotto_wasserstein import wasserstein_distance \
    as pairwise_wasserstein_distance
from scipy.ndimage import gaussian_filter
from sklearn.utils._joblib import Parallel, delayed


def betti_curves(diagrams, linspace):
    # linspace must be a three-dimensional array with the last two
    # axes having dimension equal to 1. diagrams must be a three-dimensional
    # array whose entries along axis 0 are persistence diagrams

    born = linspace >= diagrams[:, :, 0]
    not_dead = linspace < diagrams[:, :, 1]
    alive = np.logical_and(born, not_dead)
    betti = np.sum(alive, axis=2).T
    return betti


def landscapes(diagrams, linspace, n_layers):
    # Up to n_layers persistence landscapes across a collection of diagrams,
    # via sampling at regular intervals.

    n_points = diagrams.shape[1]
    n_layers_possible = min(n_points, n_layers)
    midpoints = (diagrams[:, :, 1] + diagrams[:, :, 0]) * np.sqrt(2) / 2.
    heights = (diagrams[:, :, 1] - diagrams[:, :, 0]) * np.sqrt(2) / 2.
    fibers = np.maximum(-np.abs(linspace - midpoints) + heights, 0)
    top_pos = range(n_points - n_layers_possible, n_points)
    ls = np.flip(np.partition(fibers, top_pos, axis=2)[:, :, -n_layers:],
                 axis=2)
    return np.transpose(ls, (1, 2, 0))


# def heat_function(diagram, sigma, linspace):
#     heat = np.zeros((linspace.shape[0], linspace.shape[0]))
#
#     sample_step = linspace[1] - linspace[0]
#
#     sampled_diagram = np.array(diagram // sample_step, dtype=int)
#     for sampled_point in sampled_diagram[sampled_diagram[:, 1] != 0]:
#         heat[sampled_point[0], sampled_point[1]] += 1
#         heat[sampled_point[1], sampled_point[0]] -= 1
#
#     heat = gaussian_filter(heat, sigma, mode='reflect')
#     return heat


def pairwise_betti_distances(diagrams_1, linspace, step_size,
                             diagrams_2=None, p=2., **kw_args):
    # As for pairwise_landscape_distances, but for Betti curves.

    betti_curves_1 = betti_curves(diagrams_1, linspace)
    if diagrams_2 is None:
        return (step_size ** (1 / p)) * pdist(betti_curves_1, 'minkowski', p=p)
    betti_curves_2 = betti_curves(diagrams_2, linspace)
    return (step_size ** (1 / p)) * cdist(betti_curves_1, betti_curves_2,
                                          'minkowski', p=p)


def pairwise_landscape_distances(diagrams_1, linspace, step_size,
                                 diagrams_2=None, p=2., n_layers=1, **kw_args):
    # Approximate calculation of distances between: a) if diagrams_2 is None,
    # pairs (L1, L2) of persistence landscapes both obtained from a single
    # collection diagrams_1 of diagrams; b) if diagrams_2 is not None,
    # pairs (L1, L2) of persistence landscapes such that L1 is a landscape
    # coming from diagrams_1 and L2 is a landscape coming from diagrams_2.

    n_samples_1, n_points_1 = diagrams_1.shape[:2]
    ls_1 = landscapes(diagrams_1, n_layers, linspace).reshape((
        n_samples_1, -1))
    if diagrams_2 is None:
        return (step_size ** (1 / p)) * pdist(ls_1, 'minkowski', p=p)
    n_samples_2, n_points_2 = diagrams_2.shape[:2]
    n_layers_12 = min(n_layers, n_points_1, n_points_2)
    ls_2 = landscapes(diagrams_2, n_layers_12, linspace).reshape((
        n_samples_2, -1))
    return (step_size ** (1 / p)) * cdist(ls_1, ls_2, 'minkowski', p=p)


def kernel_bottleneck_distance(diagram_x, diagram_y, delta=0.0, **kw_args):
    return pairwise_bottleneck_distance(diagram_x[diagram_x[:, 1] != 0],
                                        diagram_y[diagram_y[:, 1] != 0], delta)


def kernel_wasserstein_distance(diagram_x, diagram_y, order=1, delta=0.0,
                                **kw_args):
    return pairwise_wasserstein_distance(diagram_x, diagram_y, order, delta)


# def kernel_heat_distance(diagram_x, diagram_y, linspaces, dimension, sigma=1.,
#                          order=2, **kw_args):
#     heat_x = heat_function(diagram_x, sigma, linspaces[dimension])
#     heat_y = heat_function(diagram_y, sigma, linspaces[dimension])
#     return np.linalg.norm(heat_x - heat_y, ord=order)


implemented_metric_recipes = {'bottleneck': kernel_bottleneck_distance,
                              'wasserstein': kernel_wasserstein_distance,
                              'landscape': pairwise_landscape_distances,
                              'betti': pairwise_betti_distances,  # 'heat':
                              # kernel_heat_distance
                              }


def _parallel_pairwise(X1, metric, metric_params, X2=None, iterator=None,
                       order=None, n_jobs=None):
    metric_func = implemented_metric_recipes[metric]

    if metric in ['landscape', 'betti']:
        # Only parallelism is across dimensions here
        const_params = {key: value for key, value in metric_params.items()
                        if key not in ['linspaces', 'step_sizes']}
        lins = metric_params.get('linspaces')
        st_sizes = metric_params.get('step_sizes')
        if X2 is None:
            dist_arr = Parallel(n_jobs=n_jobs)(delayed(metric_func)(
                X1[dim], lins[dim], st_sizes[dim], **const_params)
                for dim in X1.keys())
        else:
            dist_arr = Parallel(n_jobs=n_jobs)(delayed(metric_func)(
                X1[dim], lins[dim], st_sizes[dim], diagrams_2=X2[dim],
                **const_params) for dim in X1.keys())
        dist_arr = np.stack(dist_arr, axis=2)
        if order is None:
            return dist_arr
        return np.linalg.norm(dist_arr, axis=2, ord=order)

    if iterator is None:  # TODO Remove 'heat' when this is no longer the case
        raise ValueError("iterator cannot be set to None when the metric is "
                         "'bottleneck', 'wasserstein' or 'heat'")

    n_dims = len(X1.keys())
    n_diags_1 = len(next(iter(X1.values())))
    if Y is None:
        X2, n_diags_2 = X1, n_diags_1
    else:
        n_diags_2 = len(next(iter(X2.values())))
    dist_vecs = Parallel(n_jobs=n_jobs)(delayed(metric_func)(
        X1[dim][i, :, :], X2[dim][j, :, :], dim, **metric_params)
        for i, j in iterator for dim in X1.keys())
    dist_vecs = np.array(dist_vecs).reshape((len(iterator), n_dims))
    if order is not None:
        dist_vec = np.linalg.norm(dist_vecs, axis=1, ord=order)
        dist_mat = np.zeros((n_diags_1, n_diags_2))
        dist_mat[tuple(zip(*iterator))] = dist_vec
        return dist_mat
    dist_mats = []
    for dim in X1.keys():
        dist_mat = np.zeros((n_diags_1, n_diags_2))
        dist_mat[tuple(zip(*iterator))] = dist_vecs[:, dim]
        dist_mats.append(dist_mat)
    return np.stack(dist_mats, axis=2)


def betti_amplitudes(diagrams, linspace, step_size, p=2., **kw_args):
    bcs = betti_curves(diagrams, linspace)
    return (step_size ** (1 / p)) * np.linalg.norm(bcs, axis=1, ord=p)


def landscape_amplitudes(diagrams, linspace, step_size, p=2., n_layers=1,
                         **kw_args):
    ls = landscapes(diagrams, linspace, n_layers).reshape((
        len(diagrams), -1))
    return (step_size ** (1 / p)) * np.linalg.norm(ls, axis=1, ord=p)


def bottleneck_amplitudes(diagrams, **kw_args):
    dists_to_diag = np.sqrt(2) / 2. * (diagrams[:, :, 1] - diagrams[:, :, 0])
    return np.linalg.norm(dists_to_diag, axis=1, ord=np.inf)


def wasserstein_amplitudes(diagrams, order=1., **kw_args):
    dists_to_diag = np.sqrt(2) / 2. * (diagrams[:, :, 1] - diagrams[:, :, 0])
    return np.linalg.norm(dists_to_diag, axis=1, ord=order)


# def kernel_heat_amplitude(diagram, linspaces, dimension, sigma=1., order=2,
#                           **kw_args):
#     heat = heat_function(diagram, sigma, linspaces[dimension])
#     return np.linalg.norm(heat, ord=order)


implemented_amplitude_recipes = {'bottleneck': bottleneck_amplitudes,
                                 'wasserstein': wasserstein_amplitudes,
                                 'landscape': landscape_amplitudes,
                                 'betti': betti_amplitudes,  # 'heat': kernel_heat_amplitude
                                 }


def _parallel_amplitude(X, metric, metric_params, n_jobs=None):
    amplitude_func = implemented_amplitude_recipes[metric]
    const_params = {key: value for key, value in metric_params.items()
                    if key not in ['linspaces', 'step_sizes']}
    lins = metric_params.get('linspaces')
    st_sizes = metric_params.get('step_sizes')

    # Only parallelism is across dimensions
    ampl_arr = Parallel(n_jobs=n_jobs)(delayed(amplitude_func)(
        X[dim], linspace=lins[dim], step_size=st_sizes[dim], **const_params)
        for dim in X.keys())
    ampl_arr = np.stack(ampl_arr, axis=1)
    return ampl_arr
