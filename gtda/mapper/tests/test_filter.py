import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import integers, floats
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import pdist, squareform

from gtda.mapper import Eccentricity, Entropy, Projection
from gtda.mapper.utils._list_feature_union import ListFeatureUnion
from gtda.mapper.utils.decorators import method_to_transform

from sklearn.neighbors import KernelDensity


@given(
    X=arrays(dtype=np.float,
             elements=floats(allow_nan=False,
                             allow_infinity=False),
             shape=array_shapes(min_dims=2, max_dims=2)),
    exponent=integers(min_value=1, max_value=100)
)
def test_eccentricity_shape_equals_number_of_samples(X, exponent):
    """Verify that eccentricity preserves the nb of samples in the input."""
    eccentricity = Eccentricity(exponent=exponent)
    Xt = eccentricity.fit_transform(X)
    assert Xt.shape == (len(X), 1)


@given(X=arrays(dtype=np.float,
                elements=floats(allow_nan=False,
                                allow_infinity=False),
                shape=array_shapes(min_dims=2, max_dims=2)))
def test_eccentricity_values_with_infinity_norm_equals_max_row_values(X):
    eccentricity = Eccentricity(exponent=np.inf)
    Xt = eccentricity.fit_transform(X)
    distance_matrix = squareform(pdist(X))
    assert_almost_equal(Xt, np.max(distance_matrix, axis=1).reshape(-1, 1))


@given(X=arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=-1e3,
                    max_value=-1),
    shape=array_shapes(min_dims=2, max_dims=2, min_side=2)
))
def test_entropy_values_for_negative_inputs(X):
    """Verify the numerical results of entropy (does it have the correct
    logic), on a collection of **negative** inputs."""
    entropy = Entropy()
    Xt = entropy.fit_transform(X)
    probs = X / X.sum(axis=1, keepdims=True)
    entropies = - np.einsum('ij,ij->i', probs,
                            np.where(probs != 0, np.log2(probs), 0))
    assert_almost_equal(Xt, entropies[:, None])


@given(X=arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=1,
                    max_value=1e3),
    shape=array_shapes(min_dims=2, max_dims=2, min_side=2)
))
def test_entropy_values_for_positive_inputs(X):
    """Verify the numerical results of entropy (does it have the correct logic),
    on a collection of **positive** inputs."""
    entropy = Entropy()
    Xt = entropy.fit_transform(X)
    probs = X / X.sum(axis=1, keepdims=True)
    entropies = - np.einsum('ij,ij->i', probs,
                            np.where(probs != 0, np.log2(probs), 0))
    assert_almost_equal(Xt, entropies[:, None])


@given(X=arrays(dtype=np.float,
                elements=floats(allow_nan=False,
                                allow_infinity=False),
                shape=array_shapes(min_dims=2, max_dims=2)))
def test_projection_values_equal_slice(X):
    """Test the logic of the ``Projection`` transformer."""
    columns = np.random.choice(
        X.shape[1], 1 + np.random.randint(X.shape[1]))
    Xt = Projection(columns=columns).fit_transform(X)
    assert_almost_equal(Xt, X[:, columns])


@given(X=arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=1,
                    max_value=1e3),
    shape=array_shapes(min_dims=2, max_dims=2, min_side=2),
    unique=True
))
def test_gaussian_density_values(X):
    """Check that ``fit_transform`` and ``fit + score_samples``
    of ``KernelDensity`` are the same."""
    kde_desired = KernelDensity(bandwidth=np.std(X))
    kde_actual = method_to_transform(
        KernelDensity, 'score_samples')(bandwidth=np.std(X))
    Xt_desired = kde_desired.fit(X).score_samples(X).reshape(-1, 1)
    Xt_actual = kde_actual.fit_transform(X)
    assert_almost_equal(Xt_actual, Xt_desired)


@given(X=arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=1,
                    max_value=1e3),
    shape=array_shapes(min_dims=2, max_dims=2, min_side=2),
    unique=True
))
def test_list_feature_union_transform(X):
    """Check that a ``ListFeatureUnion`` of two projections gives the same
    result as stacking the projections."""
    list_dim = [0, 1]
    p_1_2 = ListFeatureUnion([("proj" + str(k), Projection(columns=k))
                              for k in list_dim])
    p12 = Projection(columns=list_dim)
    for p in [p12, p_1_2]:
        p.fit(X)
    x_12 = p12.transform(X)
    x_1_2 = np.concatenate(p_1_2.transform(X), axis=1)

    assert_almost_equal(x_12, x_1_2)


@given(X=arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=1,
                    max_value=1e3),
    shape=array_shapes(min_dims=2, max_dims=2, min_side=2),
    unique=True
))
def test_list_feature_union_drops(X):
    """Check the the drop of ``ListFeatureUnion`` keeps the correct number
    of samples"""
    drop_0_1 = ListFeatureUnion([('drop' + str(k), 'drop')
                                 for k in range(2)])
    x_01_a = drop_0_1.fit_transform(X)
    x_01_b = drop_0_1.transform(X)
    assert x_01_a.shape == (X.shape[0], 0)
    assert x_01_b.shape == (X.shape[0], 0)
