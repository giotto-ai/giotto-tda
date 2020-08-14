"""Testing for simplicial persistent homology."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import NotFittedError

from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, \
    EuclideanCechPersistence, FlagserPersistence

pio.renderers.default = 'plotly_mimetype'

X_pc = np.array([
    [[2., 2.47942554],
     [2.47942554, 2.84147098],
     [2.98935825, 2.79848711],
     [2.79848711, 2.41211849],
     [2.41211849, 1.92484888]]
])
X_pc_list = list(X_pc)

X_dist = np.array([
    squareform(pdist(x)) for x in X_pc
])
X_dist_list = list(X_dist)

X_pc_sparse = [csr_matrix(x) for x in X_pc]
X_dist_sparse = [csr_matrix(x) for x in X_dist]


def test_vrp_params():
    metric = 'not_defined'
    vrp = VietorisRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X_pc)


def test_vrp_not_fitted():
    vrp = VietorisRipsPersistence()

    with pytest.raises(NotFittedError):
        vrp.transform(X_pc)


X_vrp_res = np.array([
    [[0., 0.43094373, 0.],
     [0., 0.5117411, 0.],
     [0., 0.60077095, 0.],
     [0., 0.62186205, 0.],
     [0.69093919, 0.80131882, 1.]]
])


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed'),
                                       (X_dist_sparse, 'precomputed')])
@pytest.mark.parametrize('max_edge_length', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_vrp_transform(X, metric, max_edge_length, infinity_values):
    vrp = VietorisRipsPersistence(max_edge_length=max_edge_length,
                                  metric=metric,
                                  infinity_values=infinity_values)
    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_res = X_vrp_res.copy()
    X_res[:, :, :2][X_res[:, :, :2] >= max_edge_length] = infinity_values
    assert_almost_equal(vrp.fit_transform(X), X_res)


def test_vrp_list_of_arrays_different_size():
    X_2 = np.array([[0., 1.], [1., 2.]])
    vrp = VietorisRipsPersistence()
    assert_almost_equal(vrp.fit_transform([X_pc[0], X_2])[0], X_vrp_res[0])


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed'),
                                       (X_dist_sparse, 'precomputed')])
def test_vrp_low_infinity_values(X, metric):
    vrp = VietorisRipsPersistence(max_edge_length=0.001,
                                  metric=metric,
                                  infinity_values=-1)
    assert_almost_equal(vrp.fit_transform(X)[:, :, :2],
                        np.zeros((1, 2, 2)))


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed'),
                                       (X_dist_sparse, 'precomputed')])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_vrp_fit_transform_plot(X, metric, hom_dims):
    VietorisRipsPersistence(metric=metric).fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


def test_srp_params():
    metric = 'not_defined'
    vrp = SparseRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X_pc)


def test_srp_not_fitted():
    srp = SparseRipsPersistence()

    with pytest.raises(NotFittedError):
        srp.transform(X_pc)


X_srp_res = np.array([
    [[0., 0.43094373, 0.],
     [0., 0.5117411, 0.],
     [0., 0.60077095, 0.],
     [0., 0.62186205, 0.],
     [0.69093919, 0.80131882, 1.]]
])


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed')])
@pytest.mark.parametrize("epsilon, diagrams",
                         [(0.0, X_vrp_res), (1.0, X_srp_res)])
def test_srp_transform(X, metric, epsilon, diagrams):
    srp = SparseRipsPersistence(metric=metric, epsilon=epsilon)

    assert_almost_equal(np.sort(srp.fit_transform(X), axis=1),
                        np.sort(diagrams, axis=1))


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed')])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_srp_fit_transform_plot(X, metric, hom_dims):
    SparseRipsPersistence(metric=metric).fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


def test_cp_params():
    coeff = 'not_defined'
    cp = EuclideanCechPersistence(coeff=coeff)

    with pytest.raises(TypeError):
        cp.fit_transform(X_pc)


def test_cp_not_fitted():
    cp = EuclideanCechPersistence()

    with pytest.raises(NotFittedError):
        cp.transform(X_pc)


X_cp_res = np.array([
    [[0., 0.31093103, 0.],
     [0., 0.30038548, 0.],
     [0., 0.25587055, 0.],
     [0., 0.21547186, 0.],
     [0.34546959, 0.41473758, 1.],
     [0.51976681, 0.55287585, 1.],
     [0.26746207, 0.28740871, 1.],
     [0.52355742, 0.52358794, 1.],
     [0.40065941, 0.40067135, 1.],
     [0.45954496, 0.45954497, 1.]]
])


@pytest.mark.parametrize('X', [X_pc, X_pc_list])
def test_cp_transform(X):
    cp = EuclideanCechPersistence()

    assert_almost_equal(cp.fit_transform(X), X_cp_res)


@pytest.mark.parametrize('X', [X_pc, X_pc_list])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_cp_fit_transform_plot(X, hom_dims):
    EuclideanCechPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


def test_fp_params():
    coeff = 'not_defined'
    fp = FlagserPersistence(coeff=coeff)

    with pytest.raises(TypeError):
        fp.fit_transform(X_dist)


def test_fp_not_fitted():
    fp = FlagserPersistence()

    with pytest.raises(NotFittedError):
        fp.transform(X_dist)


X_dir_graph = X_dist.copy()
X_dir_graph[0, 0, :] = X_dir_graph[0, 0, :] / 2.
X_dir_graph[0][np.tril_indices(5, k=-1)] = np.inf

X_dir_graph_list = [x for x in X_dir_graph]

X_dir_graph_sparse = [csr_matrix(x) for x in X_dir_graph]

X_fp_dir_res = np.array([
    [[0., 0.30038548, 0.],
     [0., 0.34546959, 0.],
     [0., 0.40065941, 0.],
     [0., 0.43094373, 0.],
     [0.5117411,  0.51976681, 1.]]
])


@pytest.mark.parametrize('X',
                         [X_dir_graph, X_dir_graph_list, X_dir_graph_sparse])
@pytest.mark.parametrize('max_edge_weight', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_fp_transform_directed(X, max_edge_weight, infinity_values):
    fp = FlagserPersistence(directed=True, max_edge_weight=max_edge_weight,
                            infinity_values=infinity_values)
    # In the undirected case with "max" filtration, the results are always the
    # same as the one of VietorisRipsPersistence
    X_res = X_fp_dir_res.copy()
    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_res[:, :, :2][X_res[:, :, :2] >= max_edge_weight] = infinity_values
    assert_almost_equal(fp.fit_transform(X), X_res)


@pytest.mark.parametrize('X', [X_dist, X_dist_list, X_dist_sparse])
@pytest.mark.parametrize('max_edge_weight', [np.inf, 0.8, 0.6])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_fp_transform_undirected(X, max_edge_weight, infinity_values):
    fp = FlagserPersistence(directed=False, max_edge_weight=max_edge_weight,
                            infinity_values=infinity_values)
    # In the undirected case with "max" filtration, the results are always the
    # same as the one of VietorisRipsPersistence
    X_res = X_vrp_res.copy()

    # In that case, subdiagrams of dimension 1 is empty
    if max_edge_weight == 0.6:
        X_res[0, -1, :] = [0., 0., 1.]

    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_res[:, :, :2][X_res[:, :, :2] >= max_edge_weight] = infinity_values
    assert_almost_equal(fp.fit_transform(X), X_res)


@pytest.mark.parametrize('X', [X_dist, X_dist_list, X_dist_sparse])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_fp_fit_transform_plot(X, hom_dims):
    FlagserPersistence(directed=False).fit_transform_plot(
        X_dist, sample=0, homology_dimensions=hom_dims)
