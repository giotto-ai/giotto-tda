"""Testing for PairwiseDistance and Amplitude"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from numpy.testing import assert_almost_equal

from gtda.diagrams import PairwiseDistance, Amplitude

X1 = np.array([
    [[0., 0.36905774, 0],
     [0., 0.37293977, 0],
     [0., 0.38995215, 0],
     [0., 3.00457644, 0],
     [0., 3.04772496, 0],
     [0., 3.32096624, 0],
     [7.97852135, 8.00382805, 1],
     [1.79289687, 1.8224113, 1],
     [1.69005811, 2.32093406, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.36905774, 0],
     [0., 0.37293977, 0],
     [0., 0.38995215, 0],
     [0., 3.00457644, 0],
     [0., 3.04772496, 0],
     [0., 3.32096624, 0],
     [15.27686119, 24.32133484, 1],
     [1.79289687, 1.8224113, 1],
     [1.69005811, 2.32093406, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 1],
     [25.24895287, 25.36620903, 2],
     [25.15629959, 25.18988037, 2],
     [25.06381798, 25.23542404, 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 1.32215321, 0],
     [0., 1.48069561, 0],
     [0., 1.62762213, 0],
     [0., 3.18582344, 0],
     [0., 3.25349188, 0],
     [0., 3.28288555, 0],
     [3.32096624, 23.92891693, 1],
     [2.93474603, 3.07139683, 1],
     [2.83503842, 2.94497037, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 1],
     [24.84643745, 25.06381798, 2],
     [24.77123451, 25.04314995, 2],
     [24.67935562, 24.93212509, 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.13842253, 0],
     [0., 0.13912651, 0],
     [0., 0.15125643, 0],
     [0., 3.30339432, 0],
     [0., 3.3078723, 0],
     [0., 3.33029914, 0],
     [16.82829285, 16.84351158, 1],
     [16.8180275, 16.84162521, 1],
     [16.80234337, 16.80629158, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.13842253, 0],
     [0., 0.13912651, 0],
     [0., 0.15125643, 0],
     [0., 2.91534185, 0],
     [0., 2.93620634, 0],
     [0., 3.00776553, 0],
     [2.85910106, 3.39503384, 1],
     [1.25564897, 1.25871313, 1],
     [1.24251938, 1.27403092, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 1],
     [3.66717434, 3.69763446, 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.13842253, 0],
     [0., 0.13912651, 0],
     [0., 0.15125643, 0],
     [0., 2.91534185, 0],
     [0., 2.93620634, 0],
     [0., 3.00776553, 0],
     [21.34438705, 24.6866188, 1],
     [2.85910106, 2.88541412, 1],
     [1.52559161, 3.39503384, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 1],
     [25.05523109, 25.27045441, 2],
     [24.93939018, 25.25673294, 2],
     [24.89836693, 25.20828819, 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]]])

X2 = np.array([
    [[0., 0.36905774, 0],
     [0., 0.37293977, 0],
     [0., 0.38995215, 0],
     [0., 3.00457644, 0],
     [0., 3.32096624, 0],
     [7.97852135, 8.00382805, 1],
     [1.79289687, 1.8224113, 1],
     [1.69005811, 2.32093406, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.36905774, 0],
     [0., 0.37293977, 0],
     [0., 3.00457644, 0],
     [0., 3.04772496, 0],
     [0., 3.32096624, 0],
     [15.27686119, 24.32133484, 1],
     [1.79289687, 1.8224113, 1],
     [1.69005811, 2.32093406, 1],
     [0., 0., 1],
     [0., 0., 1],
     [25.24895287, 25.36620903, 2],
     [25.15629959, 25.18988037, 2],
     [25.06381798, 25.23542404, 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 1.32215321, 0],
     [0., 1.48069561, 0],
     [0., 3.18582344, 0],
     [0., 3.25349188, 0],
     [0., 3.28288555, 0],
     [3.32096624, 23.92891693, 1],
     [2.93474603, 3.07139683, 1],
     [2.83503842, 2.94497037, 1],
     [0., 0., 1],
     [0., 0., 1],
     [24.84643745, 25.06381798, 2],
     [24.77123451, 25.04314995, 2],
     [24.67935562, 24.93212509, 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.13842253, 0],
     [0., 0.13912651, 0],
     [0., 0.15125643, 0],
     [0., 3.3078723, 0],
     [0., 3.33029914, 0],
     [16.82829285, 16.84351158, 1],
     [16.8180275, 16.84162521, 1],
     [16.80234337, 16.80629158, 1],
     [0., 0., 1],
     [0., 0., 1],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.13842253, 0],
     [0., 0.13912651, 0],
     [0., 0.15125643, 0],
     [0., 2.93620634, 0],
     [0., 3.00776553, 0],
     [2.85910106, 3.39503384, 1],
     [1.25564897, 1.25871313, 1],
     [1.24251938, 1.27403092, 1],
     [0., 0., 1],
     [0., 0., 1],
     [3.66717434, 3.69763446, 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2],
     [0., 0., 2]],
    [[0., 0.13842253, 0],
     [0., 0.13912651, 0],
     [0., 0.15125643, 0],
     [0., 2.93620634, 0],
     [0., 3.00776553, 0],
     [21.34438705, 24.6866188, 1],
     [2.85910106, 2.88541412, 1],
     [1.52559161, 3.39503384, 1],
     [0., 0., 1],
     [0., 0., 1],
     [25.05523109, 25.27045441, 2],
     [24.93939018, 25.25673294, 2],
     [24.89836693, 26.20828819, 2],
     [0., 0., 2],
     [0., 0., 2]]])

n_homology_dimensions = len(np.unique(X1[:, :, 2]))

X_bottleneck = np.array([
    [[0, 1, 0.],
     [0, 0, 0.],
     [0, 4, 1.]],  # Expected bottleneck ampl: [1/2, 2]

    [[0, 2, 0.],
     [0, 1, 0.],
     [0, 0, 1.]],  # Expected bottleneck ampl: [1, 0]

    [[3, 3.5, 0.],
     [0, 0, 0.],
     [5, 9, 1.]]  # Expected bottleneck ampl: [1/4, 2]
])

X_bottleneck_res_exp = np.array([
    [1/2, 2],
    [1, 0],
    [1/4, 2]
    ])


@pytest.mark.parametrize('transformer', [PairwiseDistance(), Amplitude()])
def test_not_fitted(transformer):
    with pytest.raises(NotFittedError):
        transformer.transform(X1)


parameters_distance = [
    ('bottleneck', None),
    ('wasserstein', {'p': 2, 'delta': 0.1}),
    ('betti', {'p': 2.1, 'n_bins': 10}),
    ('landscape', {'p': 2.1, 'n_bins': 10, 'n_layers': 2}),
    ('silhouette', {'p': 2.1, 'power': 1.2, 'n_bins': 10}),
    ('heat', {'p': 2.1, 'sigma': 0.5, 'n_bins': 10}),
    ('persistence_image',
     {'p': 2.1, 'sigma': 0.5, 'n_bins': 10}),
    ('persistence_image',
     {'p': 2.1, 'sigma': 0.5, 'n_bins': 10, 'weight_function': lambda x: x})
    ]


@pytest.mark.parametrize(('metric', 'metric_params'), parameters_distance)
@pytest.mark.parametrize('order', [2., None])
@pytest.mark.parametrize('n_jobs', [1, 2, -1])
def test_dd_transform(metric, metric_params, order, n_jobs):
    # X_fit == X_transform
    dd = PairwiseDistance(metric=metric, metric_params=metric_params,
                          order=order, n_jobs=n_jobs)
    X_res = dd.fit_transform(X1)
    assert (X_res.shape[0], X_res.shape[1]) == (X1.shape[0], X1.shape[0])
    if order is None:
        assert X_res.shape[2] == n_homology_dimensions

    # X_fit != X_transform
    dd = PairwiseDistance(metric=metric, metric_params=metric_params,
                          order=order, n_jobs=n_jobs)
    X_res = dd.fit(X1).transform(X2)
    assert (X_res.shape[0], X_res.shape[1]) == (X2.shape[0], X1.shape[0])
    if order is None:
        assert X_res.shape[2] == n_homology_dimensions

    # X_fit != X_transform, default metric_params
    dd = PairwiseDistance(metric=metric, order=order, n_jobs=n_jobs)
    X_res = dd.fit(X1).transform(X2)
    assert (X_res.shape[0], X_res.shape[1]) == (X2.shape[0], X1.shape[0])
    if order is None:
        assert X_res.shape[2] == n_homology_dimensions


parameters_amplitude = [
    ('bottleneck', None),
    ('wasserstein', {'p': 2}),
    ('betti', {'p': 2.1, 'n_bins': 10}),
    ('landscape', {'p': 2.1, 'n_bins': 10, 'n_layers': 2}),
    ('silhouette', {'p': 2.1, 'power': 1.2, 'n_bins': 10}),
    ('heat', {'p': 2.1, 'sigma': 0.5, 'n_bins': 10}),
    ('persistence_image',
     {'p': 2.1, 'sigma': 0.5, 'n_bins': 10}),
    ('persistence_image',
     {'p': 2.1, 'sigma': 0.5, 'n_bins': 10, 'weight_function': lambda x: x})
    ]


@pytest.mark.parametrize(('metric', 'metric_params'), parameters_amplitude)
@pytest.mark.parametrize('order', [None, 2.])
@pytest.mark.parametrize('n_jobs', [1, 2, -1])
def test_da_transform(metric, metric_params, order, n_jobs):
    n_expected_columns = n_homology_dimensions if order is None else 1

    da = Amplitude(metric=metric, metric_params=metric_params, order=order,
                   n_jobs=n_jobs)
    X_res = da.fit_transform(X1)
    assert X_res.shape == (X1.shape[0], n_expected_columns)

    # X_fit != X_transform
    da = Amplitude(metric=metric, metric_params=metric_params, order=order,
                   n_jobs=n_jobs)
    X_res = da.fit(X1).transform(X2)
    assert X_res.shape == (X2.shape[0], n_expected_columns)


@pytest.mark.parametrize(('metric', 'metric_params', 'order'),
                         [('bottleneck', None, None)])
@pytest.mark.parametrize('n_jobs', [1, 2, -1])
def test_da_transform_bottleneck(metric, metric_params, order, n_jobs):
    da = Amplitude(metric=metric, metric_params=metric_params,
                   order=order, n_jobs=n_jobs)
    X_bottleneck_res = da.fit_transform(X_bottleneck)
    assert_almost_equal(X_bottleneck_res, X_bottleneck_res_exp)


@pytest.mark.parametrize('order', [None, 2.])
@pytest.mark.parametrize('transformer_cls', [PairwiseDistance, Amplitude])
@pytest.mark.parametrize('Xnew', [X1, X2])
def test_pi_zero_weight_function(transformer_cls, order, Xnew):
    """Test that, if a zero weight function is passed to `metric_params` in
    Amplitude or PairwiseDistance when `metric` is 'persistence_image', the
    result is zero."""
    metric = 'persistence_image'
    metric_params = {
        'sigma': 0.1, 'weight_function': lambda x: x * 0., 'n_bins': 10
        }
    transformer = transformer_cls(
        metric=metric, metric_params=metric_params, order=order
        )
    X_res = transformer.fit(X1).transform(Xnew)

    assert np.array_equal(X_res, np.zeros_like(X_res))


@pytest.mark.parametrize('metric', ['heat', 'persistence_image'])
@pytest.mark.parametrize('transformer_cls', [Amplitude, PairwiseDistance])
def test_large_hk_pi_parallel(metric, transformer_cls):
    """Test that Amplitude and PairwiseDistance do not break with a read-only
    error when the input array is at least 1MB, the metric is either 'heat'
    or 'persistence_image', and more than 1 process is used (triggering
    joblib's use of memmaps)."""
    X = np.linspace(0, 100, 300000)
    n_bins = 10
    diagrams = np.expand_dims(
        np.stack([X, X, np.zeros(len(X))]).transpose(), axis=0
        )

    transformer = transformer_cls(
        metric=metric, metric_params={'sigma': 1, 'n_bins': n_bins}, n_jobs=2
        )
    Xt = transformer.fit_transform(diagrams)

    assert_almost_equal(Xt, np.zeros_like(Xt))
