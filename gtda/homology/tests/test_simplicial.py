"""Testing for simplicial persistent homology."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError
from sklearn.exceptions import NotFittedError

from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence, \
    SparseRipsPersistence, WeakAlphaPersistence, EuclideanCechPersistence, \
    FlagserPersistence

pio.renderers.default = 'plotly_mimetype'

X_pc = np.array([[[2., 2.47942554],
                  [2.47942554, 2.84147098],
                  [2.98935825, 2.79848711],
                  [2.79848711, 2.41211849],
                  [2.41211849, 1.92484888]]])
X_pc_list = list(X_pc)

X_dist = np.array([squareform(pdist(x)) for x in X_pc])
X_dist_list = list(X_dist)

X_pc_sparse = [csr_matrix(x) for x in X_pc]
X_dist_sparse = [csr_matrix(x) for x in X_dist]

X_dist_disconnected = np.array([[[0, np.inf], [np.inf, 0]]])

# 8-point sampling of a noisy circle
X_circle = np.array([[[1.00399159, -0.00797583],
                      [0.70821787, 0.68571714],
                      [-0.73369765, -0.71298056],
                      [0.01110395, -1.03739883],
                      [-0.64968271, 0.7011624],
                      [0.03895963, 0.94494511],
                      [0.76291108, -0.68774373],
                      [-1.01932365, -0.05793851]]])

# 3d point cloud example -- same as 2d example but z-coord is 0
X_pc_3d = np.array([[[2., 2.47942554, 0.],
                    [2.47942554, 2.84147098, 0.],
                    [2.98935825, 2.79848711, 0.],
                    [2.79848711, 2.41211849, 0.],
                    [2.41211849, 1.92484888, 0.]]])


def test_vrp_params():
    metric = 'not_defined'
    vrp = VietorisRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X_pc)


def test_vrp_not_fitted():
    vrp = VietorisRipsPersistence()

    with pytest.raises(NotFittedError):
        vrp.transform(X_pc)


X_vrp_exp = np.array([[[0., 0.43094373, 0.],
                       [0., 0.5117411, 0.],
                       [0., 0.60077095, 0.],
                       [0., 0.62186205, 0.],
                       [0.69093919, 0.80131882, 1.]]])


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed'),
                                       (X_dist_sparse, 'precomputed')])
@pytest.mark.parametrize('collapse_edges', [True, False])
@pytest.mark.parametrize('max_edge_length', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_vrp_transform(X, metric, collapse_edges, max_edge_length,
                       infinity_values):
    vrp = VietorisRipsPersistence(metric=metric,
                                  collapse_edges=collapse_edges,
                                  max_edge_length=max_edge_length,
                                  infinity_values=infinity_values)
    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_exp = X_vrp_exp.copy()
    X_exp[:, :, :2][X_exp[:, :, :2] >= max_edge_length] = infinity_values
    assert_almost_equal(vrp.fit_transform(X), X_exp)


def test_vrp_list_of_arrays_different_size():
    X_2 = np.array([[0., 1.], [1., 2.]])
    vrp = VietorisRipsPersistence()
    assert_almost_equal(vrp.fit_transform([X_pc[0], X_2])[0], X_vrp_exp[0])


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
                                       (X_dist_disconnected, 'precomputed')])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_vrp_fit_transform_plot(X, metric, hom_dims):
    VietorisRipsPersistence(metric=metric).fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims
        )


def test_wrp_params():
    metric = 'not_defined'
    wrp = WeightedRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        wrp.fit_transform(X_pc)


def test_wrp_metric_params():
    def metric(x, y, **kwargs):
        return np.linalg.norm(x - y)

    metric_params = {"parameter": 0.}
    wrp = WeightedRipsPersistence(metric=metric, metric_params=metric_params)
    wrp.fit_transform(X_pc)


def test_wrp_not_fitted():
    wrp = WeightedRipsPersistence()

    with pytest.raises(NotFittedError):
        wrp.transform(X_pc)


def test_wrp_notimplemented_string_weights():
    wrp = WeightedRipsPersistence(weights="foo")

    with pytest.raises(ValueError, match="'foo' passed for `weights` but the "
                                         "only allowed string is 'DTM'"):
        wrp.fit(X_pc)


def test_wrp_notimplemented_p():
    wrp = WeightedRipsPersistence(weight_params={'p': 1.2})

    with pytest.raises(ValueError):
        wrp.fit(X_pc)


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed'),
                                       (X_dist_sparse, 'precomputed')])
@pytest.mark.parametrize('weight_params', [{'p': 1}, {'p': 2}, {'p': np.inf}])
@pytest.mark.parametrize('collapse_edges', [True, False])
@pytest.mark.parametrize('max_edge_weight', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_wrp_same_as_vrp_when_zero_weights(X, metric, weight_params,
                                           collapse_edges, max_edge_weight,
                                           infinity_values):
    wrp = WeightedRipsPersistence(weights=lambda x: np.zeros(x.shape[0]),
                                  weight_params=weight_params,
                                  metric=metric,
                                  collapse_edges=collapse_edges,
                                  max_edge_weight=max_edge_weight,
                                  infinity_values=infinity_values)

    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_exp = X_vrp_exp.copy()
    X_exp[:, :, :2][X_exp[:, :, :2] >= max_edge_weight] = infinity_values
    assert_almost_equal(wrp.fit_transform(X), X_exp)


X_wrp_exp = {1: np.array([[[0.95338798, 1.474913, 0.],
                           [1.23621261, 1.51234496, 0.],
                           [1.21673107, 1.68583047, 0.],
                           [1.30722439, 1.73876917, 0.],
                           [0., 0., 1.]]]),
             2: np.array([[[0.95338798, 1.08187652, 0.],
                           [1.23621261, 1.2369417, 0.],
                           [1.21673107, 1.26971364, 0.],
                           [1.30722439, 1.33688354, 0.],
                           [0., 0., 1.]]]),
             np.inf: np.array([[[0., 0., 0.],
                                [0., 0., 1.]]])}


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed'),
                                       (X_dist_sparse, 'precomputed')])
@pytest.mark.parametrize('weight_params', [{'p': 1}, {'p': 2}, {'p': np.inf}])
@pytest.mark.parametrize('collapse_edges', [True, False])
def test_wrp_transform(X, metric, weight_params, collapse_edges):
    wrp = WeightedRipsPersistence(weight_params=weight_params,
                                  metric=metric,
                                  collapse_edges=collapse_edges)

    assert_almost_equal(wrp.fit_transform(X), X_wrp_exp[weight_params['p']])


def test_wrp_infinity_error():
    with pytest.raises(ValueError, match="Input contains"):
        wrp = WeightedRipsPersistence(metric='precomputed')
        wrp.fit_transform(X_dist_disconnected)


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean')])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_wrp_fit_transform_plot(X, metric, hom_dims):
    WeightedRipsPersistence(
        metric=metric, weight_params={'n_neighbors': 1}
        ).fit_transform_plot(X, sample=0, homology_dimensions=hom_dims)


def test_srp_params():
    metric = 'not_defined'
    vrp = SparseRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X_pc)


def test_srp_not_fitted():
    srp = SparseRipsPersistence()

    with pytest.raises(NotFittedError):
        srp.transform(X_pc)


X_srp_exp = np.array([[[0., 0.43094373, 0.],
                       [0., 0.5117411, 0.],
                       [0., 0.60077095, 0.],
                       [0., 0.62186205, 0.],
                       [0.69093919, 0.80131882, 1.]]])


@pytest.mark.parametrize('X, metric', [(X_pc, 'euclidean'),
                                       (X_pc_list, 'euclidean'),
                                       (X_pc_sparse, 'euclidean'),
                                       (X_dist, 'precomputed'),
                                       (X_dist_list, 'precomputed')])
@pytest.mark.parametrize("epsilon, diagrams",
                         [(0.0, X_vrp_exp), (1.0, X_srp_exp)])
def test_srp_transform(X, metric, epsilon, diagrams):
    srp = SparseRipsPersistence(metric=metric, epsilon=epsilon)

    assert_almost_equal(np.sort(srp.fit_transform(X), axis=1),
                        np.sort(diagrams, axis=1))


@pytest.mark.parametrize('X', [X_pc, X_pc_list])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_srp_fit_transform_plot(X, hom_dims):
    SparseRipsPersistence().fit_transform_plot(X, sample=0,
                                               homology_dimensions=hom_dims)


def test_wap_params():
    coeff = 'not_defined'
    wap = WeakAlphaPersistence(coeff=coeff)

    with pytest.raises(TypeError):
        wap.fit_transform(X_pc)


def test_wap_not_fitted():
    wap = WeakAlphaPersistence()

    with pytest.raises(NotFittedError):
        wap.transform(X_pc)


# On this particular X_pc, WeakAlpha and VietorisRips should give the exact
# same result
X_wap_exp = X_vrp_exp


@pytest.mark.parametrize('X', [X_pc, X_pc_list])
@pytest.mark.parametrize('max_edge_length', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_wap_transform(X, max_edge_length, infinity_values):
    wap = WeakAlphaPersistence(max_edge_length=max_edge_length,
                               infinity_values=infinity_values)
    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_exp = X_wap_exp.copy()
    X_exp[:, :, :2][X_exp[:, :, :2] >= max_edge_length] = infinity_values
    assert_almost_equal(wap.fit_transform(X), X_exp)


@pytest.mark.parametrize("transformer_cls", [VietorisRipsPersistence,
                                             WeakAlphaPersistence])
def test_vrp_wap_transform_circle(transformer_cls):
    """Test that, on a sampled noisy circle, both VietorisRipsPersistence and
    WeakAlphaPersistence lead to reasonable barcodes"""
    transformer = transformer_cls()
    X_res = transformer.fit_transform(X_circle)
    subdiagram_0 = X_res[X_res[:, :, 2] == 0]
    subdiagram_1 = X_res[X_res[:, :, 2] == 1]
    length_reg_pol = 2 * np.sin(np.pi / X_circle.shape[1])
    last_conn_comp_param = np.max(subdiagram_0[:, 1])
    assert last_conn_comp_param < length_reg_pol + 0.1
    assert len(subdiagram_1) == 1
    assert subdiagram_1[0, 0] > last_conn_comp_param
    assert subdiagram_1[0, 1] > np.sqrt(3)


def test_wap_qhullerror():
    """"Test that SciPy raises a QhullError when there are too few points (at
    least 4 are needed)"""
    X_pc_2 = np.array([[[0., 1.], [1., 2.], [2., 3.]]])
    wap = WeakAlphaPersistence()
    with pytest.raises(QhullError):
        wap.fit_transform(X_pc_2)


def test_wap_list_of_arrays_different_size():
    X = [X_pc[0], X_pc[0][:-1]]
    wap = WeakAlphaPersistence()
    assert_almost_equal(wap.fit_transform(X)[0], X_wap_exp[0])


@pytest.mark.parametrize('X', [X_pc, X_pc_list])
def test_wap_low_infinity_values(X):
    wap = WeakAlphaPersistence(max_edge_length=0.001, infinity_values=-1)
    assert_almost_equal(wap.fit_transform(X)[:, :, :2],
                        np.zeros((1, 2, 2)))


@pytest.mark.parametrize('X', [X_pc, X_pc_list])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_wap_fit_transform_plot(X, hom_dims):
    WeakAlphaPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims
        )


def test_cp_params():
    coeff = 'not_defined'
    cp = EuclideanCechPersistence(coeff=coeff)

    with pytest.raises(TypeError):
        cp.fit_transform(X_pc)


def test_cp_not_fitted():
    cp = EuclideanCechPersistence()

    with pytest.raises(NotFittedError):
        cp.transform(X_pc)


X_cp_exp = np.array([[[0., 0.31093103, 0.],
                      [0., 0.30038548, 0.],
                      [0., 0.25587055, 0.],
                      [0., 0.21547186, 0.],
                      [0.34546959, 0.41473758, 1.],
                      [0.51976681, 0.55287585, 1.],
                      [0.26746207, 0.28740871, 1.],
                      [0.52355742, 0.52358794, 1.],
                      [0.40065941, 0.40067135, 1.],
                      [0.45954496, 0.45954497, 1.]]])


@pytest.mark.parametrize('X', [X_pc, X_pc_list, X_pc_3d])
def test_cp_transform(X):
    cp = EuclideanCechPersistence()

    assert_almost_equal(cp.fit_transform(X), X_cp_exp)


@pytest.mark.parametrize('X', [X_pc, X_pc_list])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_cp_fit_transform_plot(X, hom_dims):
    EuclideanCechPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims
        )


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

X_fp_dir_exp = np.array([[[0., 0.30038548, 0.],
                          [0., 0.34546959, 0.],
                          [0., 0.40065941, 0.],
                          [0., 0.43094373, 0.],
                          [0.5117411,  0.51976681, 1.]]])


@pytest.mark.parametrize('X',
                         [X_dir_graph, X_dir_graph_list, X_dir_graph_sparse])
@pytest.mark.parametrize('max_edge_weight', [np.inf, 0.8])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_fp_transform_directed(X, max_edge_weight, infinity_values):
    fp = FlagserPersistence(directed=True, max_edge_weight=max_edge_weight,
                            infinity_values=infinity_values)
    # In the undirected case with "max" filtration, the results are always the
    # same as the one of VietorisRipsPersistence
    X_exp = X_fp_dir_exp.copy()
    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_exp[:, :, :2][X_exp[:, :, :2] >= max_edge_weight] = infinity_values
    assert_almost_equal(fp.fit_transform(X), X_exp)


@pytest.mark.parametrize('X', [X_dist, X_dist_list, X_dist_sparse])
@pytest.mark.parametrize('max_edge_weight', [np.inf, 0.8, 0.6])
@pytest.mark.parametrize('infinity_values', [10, 30])
def test_fp_transform_undirected(X, max_edge_weight, infinity_values):
    fp = FlagserPersistence(directed=False, max_edge_weight=max_edge_weight,
                            infinity_values=infinity_values)
    # In the undirected case with "max" filtration, the results are always the
    # same as the one of VietorisRipsPersistence
    X_exp = X_vrp_exp.copy()

    # In that case, the subdiagram of dimension 1 is empty
    if max_edge_weight == 0.6:
        X_exp[0, -1, :] = [0., 0., 1.]

    # This is not generally true, it is only a way to obtain the res array
    # in this specific case
    X_exp[:, :, :2][X_exp[:, :, :2] >= max_edge_weight] = infinity_values
    assert_almost_equal(fp.fit_transform(X), X_exp)


@pytest.mark.parametrize('delta', range(1, 4))
def test_fp_transform_high_hom_dim(delta):
    """Test that if the maximum homology dimension is greater than or equal to
    the number of points, we do not produce errors."""
    n_points = 3
    X = X_dist[:, :n_points, :n_points]
    fp = FlagserPersistence(homology_dimensions=list(range(n_points + delta)))
    assert_almost_equal(fp.fit_transform(X)[0, -1],
                        np.array([0., 0., n_points + delta - 1], dtype=float))


@pytest.mark.parametrize('X', [X_dist, X_dist_list, X_dist_disconnected])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_fp_fit_transform_plot(X, hom_dims):
    FlagserPersistence(directed=False).fit_transform_plot(
        X_dist, sample=0, homology_dimensions=hom_dims
        )
