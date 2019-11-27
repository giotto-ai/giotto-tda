import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import integers
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import pdist, squareform

from giotto.mapper import Eccentricity


@given(
    X=arrays(dtype=np.float, shape=array_shapes(min_dims=2, max_dims=2)),
    exponent=integers(min_value=1, max_value=100)
)
def test_eccentricity_shape_equals_number_of_samples(X, exponent):
    eccentricity = Eccentricity(exponent=exponent)
    Xt = eccentricity.fit_transform(X)
    assert Xt.shape == (len(X),)


@given(X=arrays(dtype=np.float, shape=array_shapes(min_dims=2, max_dims=2)))
def test_eccentricity_values_with_infinity_norm_equals_max_row_values(X):
    eccentricity = Eccentricity(exponent=np.inf)
    Xt = eccentricity.fit_transform(X)
    distance_matrix = squareform(pdist(X))
    assert_almost_equal(Xt, np.max(distance_matrix, axis=1))
