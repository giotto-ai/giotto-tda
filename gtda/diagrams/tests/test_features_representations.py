"""Testing for features and vector representations."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, integers
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.diagrams import PersistenceEntropy, BettiCurve, \
    PersistenceLandscape, HeatKernel, PersistenceImage, Silhouette

pio.renderers.default = 'plotly_mimetype'

X = np.array([[[0., 1., 0.], [2., 3., 0.], [4., 6., 1.], [2., 6., 1.]]])

line_plots_traces_params = {"mode": "lines+markers"}
heatmap_trace_params = {"colorscale": "viridis"}
layout_params = {"title": "New title"}


@pytest.mark.parametrize('transformer',
                         [PersistenceEntropy(), BettiCurve(),
                          PersistenceLandscape(), HeatKernel(),
                          PersistenceImage(), Silhouette()])
def test_not_fitted(transformer):
    with pytest.raises(NotFittedError):
        transformer.transform(X)


@pytest.mark.parametrize('transformer',
                         [HeatKernel(), PersistenceImage()])
@pytest.mark.parametrize('hom_dim_idx', [0, 1])
def test_fit_transform_plot_one_hom_dim(transformer, hom_dim_idx):
    plotly_params = \
        {"trace": heatmap_trace_params, "layout": layout_params}
    transformer.fit_transform_plot(
        X, sample=0, homology_dimension_idx=hom_dim_idx,
        plotly_params=plotly_params
        )


@pytest.mark.parametrize('transformer',
                         [BettiCurve(), PersistenceLandscape(), Silhouette()])
@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_fit_transform_plot_many_hom_dims(transformer, hom_dims):
    plotly_params = \
        {"traces": line_plots_traces_params, "layout": layout_params}
    transformer.fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims, plotly_params=plotly_params
        )


@pytest.mark.parametrize('transformer',
                         [HeatKernel(), PersistenceImage(), BettiCurve(),
                          PersistenceLandscape(), Silhouette()])
def test_fit_transform_plot_infinite_hom_dims(transformer):
    X_infinite_hom_dim = X.copy()
    X_infinite_hom_dim[:, :, 2] = np.inf
    transformer.fit_transform_plot(X_infinite_hom_dim, sample=0)


@pytest.mark.parametrize('transformer',
                         [BettiCurve(), PersistenceLandscape(), Silhouette()])
def test_fit_transform_plot_wrong_hom_dims(transformer):
    with pytest.raises(ValueError):
        transformer.fit_transform_plot(X, sample=0, homology_dimensions=(2,))


def test_pe_transform():
    pe = PersistenceEntropy()
    diagram_res = np.array([[1., 0.91829583405]])

    assert_almost_equal(pe.fit_transform(X), diagram_res)

    pe_normalize = PersistenceEntropy(normalize=True)
    diagram_res = np.array([[1., 0.355245321276]])
    assert_almost_equal(pe_normalize.fit_transform(X), diagram_res)


@pytest.mark.parametrize('n_bins', list(range(10, 51, 10)))
def test_bc_transform_shape(n_bins):
    bc = BettiCurve(n_bins=n_bins)
    X_res = bc.fit_transform(X)
    assert X_res.shape == (1, bc._n_dimensions, n_bins)


@pytest.mark.parametrize('n_bins', list(range(10, 51, 10)))
@pytest.mark.parametrize('n_layers', list(range(1, 10)))
def test_pl_transform_shape(n_bins, n_layers):
    pl = PersistenceLandscape(n_bins=n_bins, n_layers=n_layers)
    X_res = pl.fit_transform(X)
    assert X_res.shape == (1, pl._n_dimensions, n_layers, n_bins)


def test_pi_zero_weight_function():
    pi = PersistenceImage(weight_function=lambda x: x * 0.)
    X_res = pi.fit_transform(X)
    assert np.array_equal(X_res, np.zeros_like(X_res))


@given(X=arrays(dtype=np.float, unique=True,
                elements=floats(min_value=-10, max_value=10),
                shape=array_shapes(min_dims=1, max_dims=1, min_side=11)))
def test_pi_null(X):
    """Test that, if one trivial diagram (all pts on the diagonal) is provided,
    along with a non-trivial one, then its persistence image is null"""
    n_bins = 10
    X = np.append(X, 1 + X[-1])
    diagrams = np.expand_dims(
        np.stack([X, X, np.zeros(len(X))]).transpose(), axis=0
        )
    diagrams = np.repeat(diagrams, 2, axis=0)
    diagrams[1, :, 1] += 1

    sigma = (np.max(diagrams[:, :, 1] - np.min(diagrams[:, :, 0]))) / 2
    pi = PersistenceImage(sigma=sigma, n_bins=n_bins)

    assert_almost_equal(pi.fit_transform(diagrams)[0], 0)


@given(pts=arrays(dtype=np.float, unique=True,
                  elements=floats(allow_nan=False,
                                  allow_infinity=False,
                                  min_value=-10,
                                  max_value=10),
                  shape=(20, 2)))
def test_pi_positive(pts):
    diagrams = np.expand_dims(
        np.concatenate([np.sort(pts, axis=1), np.zeros((pts.shape[0], 1))],
                       axis=1),
        axis=0
        )
    sigma = (np.max(diagrams[:, :, 1] - np.min(diagrams[:, :, 0]))) / 2
    pi = PersistenceImage(sigma=sigma)
    assert np.all(pi.fit_transform(diagrams) >= 0.)


def test_large_pi_null_multithreaded():
    """Test that pi is computed correctly when the input array is at least 1MB
    and more than 1 process is used, triggering joblib's use of memmaps"""
    X = np.linspace(0, 100, 300000)
    pi = PersistenceImage(sigma=1, n_bins=10, n_jobs=2)
    diagrams = np.expand_dims(
        np.stack([X, X, np.zeros(len(X))]).transpose(), axis=0
        )
    diagrams = np.repeat(diagrams, 2, axis=0)
    diagrams[1, :, 1] += 1

    assert_almost_equal(pi.fit_transform(diagrams)[0], 0)


def test_silhouette_transform():
    sht = Silhouette(n_bins=31, power=1.)
    X_sht_res = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.1,
                          0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 0.05,
                          0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.])

    assert_almost_equal(sht.fit_transform(X)[0][0], X_sht_res)


def test_silhouette_big_order():
    diagrams = np.array([[[0, 2, 0], [1, 4, 0]]])
    sht_10 = Silhouette(n_bins=41, power=10.)
    X_sht_res = np.array([0., 0.00170459, 0.00340919, 0.00511378, 0.00681837,
                          0.00852296, 0.01022756, 0.01193215, 0.01363674,
                          0.01534133, 0.01704593, 0.11363674, 0.21022756,
                          0.30681837, 0.40340919, 0.5, 0.59659081, 0.69318163,
                          0.78977244, 0.88636326, 0.98295407, 1.08124948,
                          1.17954489, 1.27784029, 1.3761357, 1.47443111,
                          1.3761357, 1.27784029, 1.17954489, 1.08124948,
                          0.98295407, 0.88465867, 0.78636326, 0.68806785,
                          0.58977244, 0.49147704, 0.39318163, 0.29488622,
                          0.19659081, 0.09829541, 0.])

    assert_almost_equal(sht_10.fit_transform(diagrams)[0][0], X_sht_res)


@pytest.mark.parametrize('transformer', [HeatKernel(), PersistenceImage()])
def test_all_pts_the_same(transformer):
    X = np.zeros((1, 4, 3))
    X_res = transformer.fit_transform(X)
    assert np.array_equal(X_res, np.zeros_like(X_res))


pts_gen = arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=1.,
                    max_value=10.),
    shape=(1, 20, 2), unique=True
    )
dims_gen = arrays(
    dtype=np.int,
    elements=integers(min_value=0,
                      max_value=3),
    shape=(1, 20, 1)
    )


def _validate_distinct(X):
    """Check if, in X, there is any persistence diagram for which all births
    and deaths are equal."""
    unique_values = [np.unique(x[:, 0:2]) for x in X]
    if np.any([len(u) < 2 for u in unique_values]):
        raise ValueError
    return 0


def get_input(pts, dims):
    for p in pts:
        try:
            _validate_distinct([pts])
        except ValueError:
            p[0, 0:2] += 0.3
            # add a distinct value, if not provided by hypothesis
    X = np.concatenate([np.sort(pts, axis=2), dims], axis=2)
    return X


@given(pts_gen, dims_gen)
def test_hk_shape(pts, dims):
    n_bins = 10
    X = get_input(pts, dims)
    sigma = (np.max(X[:, :, :2]) - np.min(X[:, :, :2])) / 2

    hk = HeatKernel(sigma=sigma, n_bins=n_bins)
    num_dimensions = len(np.unique(dims))
    X_t = hk.fit_transform(X)

    assert X_t.shape == (X.shape[0], num_dimensions, n_bins, n_bins)


@given(pts_gen, dims_gen)
def test_hk_positive(pts, dims):
    """We expect the points above the PD-diagonal to be non-negative (up to a
    numerical error)"""
    n_bins = 10
    X = get_input(pts, dims)
    sigma = (np.max(X[:, :, :2]) - np.min(X[:, :, :2])) / 2

    hk = HeatKernel(sigma=sigma, n_bins=n_bins)
    X_t = hk.fit_transform(X)

    assert np.all((np.tril(X_t[:, :, ::-1, :]) + 1e-13) >= 0.)


@pytest.mark.parametrize('transformer_cls', [HeatKernel, PersistenceImage])
@given(pts_gen, dims_gen)
def test_hk_pi_big_sigma(transformer_cls, pts, dims):
    """We expect that with a huge sigma, the diagrams are so diluted that they
    are almost 0. Effectively, verifies that the smoothing is applied."""
    n_bins = 10
    X = get_input(pts, dims)
    # To make the test less flaky, it helps to set al homology dimensions equal
    X[:, :, 2] = 0.
    sigma = 100 * (np.max(X[:, :, :2]) - np.min(X[:, :, :2]))

    hk = transformer_cls(sigma=sigma, n_bins=n_bins)
    X_t = hk.fit_transform(X)

    max_hk_abs_value = np.max(np.abs(X_t))
    assert max_hk_abs_value <= 1e-4


@given(pts_gen)
def test_hk_with_diag_points(pts):
    """Add points on the diagonal, and verify that we have the same results
    (on the same fitted values)."""
    n_bins = 10
    hk = HeatKernel(sigma=1, n_bins=n_bins)

    X = get_input(pts, np.zeros((pts.shape[0], pts.shape[1], 1)))
    diag_points = np.array([[[2, 2, 0], [3, 3, 0], [7, 7, 0]]])
    X_with_diag_points = np.concatenate([X, diag_points], axis=1)

    hk = hk.fit(X_with_diag_points)

    X_t, X_with_diag_points_t = [hk.transform(X_)
                                 for X_ in [X, X_with_diag_points]]

    assert_almost_equal(X_with_diag_points_t, X_t, decimal=13)


def test_large_hk_shape_multithreaded():
    """Test that HeatKernel returns something of the right shape when the input
    array is at least 1MB and more than 1 process is used, triggering joblib's
    use of memmaps"""
    X = np.linspace(0, 100, 300000)
    n_bins = 10
    diagrams = np.expand_dims(
        np.stack([X, X, np.zeros(len(X))]).transpose(), axis=0
        )

    hk = HeatKernel(sigma=1, n_bins=n_bins, n_jobs=2)
    num_dimensions = 1
    x_t = hk.fit_transform(diagrams)

    assert x_t.shape == (diagrams.shape[0], num_dimensions, n_bins, n_bins)
