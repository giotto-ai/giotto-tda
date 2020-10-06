"""Testing of preprocessing tools for persistence diagrams."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.diagrams import ForgetDimension, Scaler, Filtering

pio.renderers.default = 'plotly_mimetype'
plotly_params = {"trace": {"marker_size": 20},
                 "layout": {"title": "New title"}}

X_1 = np.array([[[0., 0.36905774, 0],
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

X_2 = np.array([[[0., 0.36905774, 0],
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


def test_not_fitted():
    with pytest.raises(NotFittedError):
        ForgetDimension().transform(X_1)

    with pytest.raises(NotFittedError):
        Scaler().transform(X_1)

    with pytest.raises(NotFittedError):
        Scaler().inverse_transform(X_1)

    with pytest.raises(NotFittedError):
        Filtering().transform(X_1)


def test_forg_fit_transform_plot():
    ForgetDimension().fit_transform_plot(
        X_1, sample=0, plotly_params=plotly_params
    )


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,)])
def test_fit_transform_plot(hom_dims):
    Scaler().fit_transform_plot(
        X_1, sample=0, homology_dimensions=hom_dims,
        plotly_params=plotly_params
    )

    Filtering().fit_transform_plot(
        X_1, sample=0, homology_dimensions=hom_dims,
        plotly_params=plotly_params
    )


@pytest.mark.parametrize('X', [X_1, X_2])
def test_forg_transform_shape(X):
    forg = ForgetDimension()
    X_res = forg.fit_transform(X)
    assert X_res.shape == X.shape


parameters_sc = [
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


@pytest.mark.parametrize(('metric', 'metric_params'), parameters_sc)
@pytest.mark.parametrize('X', [X_1, X_2])
def test_sc_transform_shape(X, metric, metric_params):
    sc = Scaler(metric=metric, metric_params=metric_params, n_jobs=1)
    X_res = sc.fit_transform(X)
    assert X_res.shape == X.shape

    X_inv_res = sc.inverse_transform(X_res)
    assert_almost_equal(X_inv_res, X)


@pytest.mark.parametrize('X', [X_1, X_2])
def test_filt_transform_zero(X):
    filt = Filtering(epsilon=0.)
    X_res = filt.fit_transform(X[:, [0], :])
    assert_almost_equal(X_res, X[:, [0], :])


def total_lifetimes_in_dims(X, dims):
    return sum([
        np.sum(np.diff(X[X[:, :, 2] == dim], axis=1)[:, 0]) for dim in dims
        ])


@pytest.mark.parametrize('homology_dimensions', [None, (0, 1, 2), (0,), (1,),
                                                 (2,), (0, 1), (0, 2), (1, 2)])
def test_filt_transform_unfiltered_hom_dims(homology_dimensions):
    filt = Filtering(epsilon=2., homology_dimensions=homology_dimensions)
    X_1_res = filt.fit_transform(X_1)
    if homology_dimensions is None:
        unfiltered_hom_dims = []
    else:
        unfiltered_hom_dims = [
            dim for dim in filt.homology_dimensions_
            if dim not in homology_dimensions
            ]
    assert total_lifetimes_in_dims(X_1, unfiltered_hom_dims) == \
           total_lifetimes_in_dims(X_1_res, unfiltered_hom_dims)


lifetimes_1 = X_1[:, :, 1] - X_1[:, :, 0]
epsilons_1 = np.linspace(np.min(lifetimes_1), np.max(lifetimes_1), num=3)


@pytest.mark.parametrize('epsilon', epsilons_1)
def test_filt_transform(epsilon):
    filt = Filtering(epsilon=epsilon)
    X_res_1 = filt.fit_transform(X_1)
    assert X_res_1.shape[1] <= X_1.shape[1]

    lifetimes_res_1 = X_res_1[:, :, 1] - X_res_1[:, :, 0]
    assert not ((lifetimes_res_1 > 0.) & (lifetimes_res_1 <= epsilon)).any()
