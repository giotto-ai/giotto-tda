"""Testing for simplicial persistent homology."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import NotFittedError

from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, \
    EuclideanCechPersistence, FlagserPersistence

pio.renderers.default = 'plotly_mimetype'

X = np.array([[[2., 2.47942554], [2.47942554, 2.84147098],
               [2.98935825, 2.79848711], [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])

X_dist = squareform(pdist(X[0])).reshape(*X.shape[:2], X.shape[1])


def test_vrp_params():
    metric = 'not_defined'
    vrp = VietorisRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X)


def test_vrp_not_fitted():
    vrp = VietorisRipsPersistence()

    with pytest.raises(NotFittedError):
        vrp.transform(X)


X_vrp_res = np.array([[[0., 0.43094373, 0], [0., 0.5117411, 0],
                       [0., 0.60077095, 0], [0., 0.62186205, 0],
                       [0.69093919, 0.80131882, 1]]])


@pytest.mark.parametrize('max_edge_length', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_vrp_transform(max_edge_length, infinity_values):
    vrp = VietorisRipsPersistence(max_edge_length=max_edge_length,
                                  infinity_values=infinity_values)
    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_res = X_vrp_res.copy()
    X_res[:, :, :2][X_res[:, :, :2] >= max_edge_length] = infinity_values
    assert_almost_equal(vrp.fit_transform(X), X_res)


def test_vrp_list_of_arrays():
    X_2 = np.array([[0., 1.], [1., 2.]])
    X_list = [X[0].copy(), X_2]
    vrp = VietorisRipsPersistence()
    vrp.fit(X_list)


def test_vrp_low_infinty_values():
    vrp = VietorisRipsPersistence(max_edge_length=0.001,
                                  infinity_values=-1)
    assert_almost_equal(vrp.fit_transform(X)[:, :, :2],
                        np.zeros((1, 2, 2)))


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_vrp_fit_transform_plot(hom_dims):
    VietorisRipsPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


def test_srp_params():
    metric = 'not_defined'
    vrp = SparseRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X)


def test_srp_not_fitted():
    srp = SparseRipsPersistence()

    with pytest.raises(NotFittedError):
        srp.transform(X)


X_srp_res = np.array([[[0., 0.43094373, 0], [0., 0.5117411, 0],
                       [0., 0.60077095, 0], [0., 0.62186205, 0],
                       [0.69093919, 0.80131882, 1]]])


@pytest.mark.parametrize("epsilon, point_clouds, diagrams",
                         [(0.0, X, X_vrp_res),
                          (1.0, X, X_srp_res)])
def test_srp_transform(epsilon, point_clouds, diagrams):
    srp = SparseRipsPersistence(epsilon=epsilon)

    assert_almost_equal(np.sort(srp.fit_transform(point_clouds), axis=1),
                        np.sort(diagrams, axis=1))


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_srp_fit_transform_plot(hom_dims):
    SparseRipsPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


def test_cp_params():
    coeff = 'not_defined'
    cp = EuclideanCechPersistence(coeff=coeff)

    with pytest.raises(TypeError):
        cp.fit_transform(X)


def test_cp_not_fitted():
    cp = EuclideanCechPersistence()

    with pytest.raises(NotFittedError):
        cp.transform(X)


X_cp_res = np.array(
    [[[0., 0.31093103, 0.], [0., 0.30038548, 0.],
      [0., 0.25587055, 0.], [0., 0.21547186, 0.],
      [0.34546959, 0.41473758, 1.], [0.51976681, 0.55287585, 1.],
      [0.26746207, 0.28740871, 1.], [0.52355742, 0.52358794, 1.],
      [0.40065941, 0.40067135, 1.], [0.45954496, 0.45954497, 1.]]])


def test_cp_transform():
    cp = EuclideanCechPersistence()

    assert_almost_equal(cp.fit_transform(X), X_cp_res)


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_cp_fit_transform_plot(hom_dims):
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

X_fp_dir_res = np.array([[[0., 0.30038548, 0.], [0., 0.34546959, 0.],
                          [0., 0.40065941, 0.], [0., 0.43094373, 0.],
                          [0.5117411,  0.51976681, 1.]]])


@pytest.mark.parametrize('max_edge_weight', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_fp_transform_directed(max_edge_weight, infinity_values):
    fp = FlagserPersistence(directed=True, max_edge_weight=max_edge_weight,
                            infinity_values=infinity_values)
    # In the undirected case with "max" filtration, the results are always the
    # same as the one of VietorisRipsPersistence
    X_res = X_fp_dir_res.copy()
    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_res[:, :, :2][X_res[:, :, :2] >= max_edge_weight] = infinity_values
    assert_almost_equal(fp.fit_transform(X_dir_graph), X_res)


@pytest.mark.parametrize('max_edge_weight', [np.inf, 0.8, 0.6])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_fp_transform_undirected(max_edge_weight, infinity_values):
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
    assert_almost_equal(fp.fit_transform(X_dist), X_res)


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_fp_fit_transform_plot(hom_dims):
    FlagserPersistence(directed=False).fit_transform_plot(
        X_dist, sample=0, homology_dimensions=hom_dims)
