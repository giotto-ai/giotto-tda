"""Testing for filter utils."""
# License : Apache 2.0

import numpy as np
from giotto.mapper import Eccentricity
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import pdist, squareform


@given(X=arrays(dtype=np.float, shape=array_shapes(min_dims=2, max_dims=2)))
def test_eccentricity_shape_equals_number_of_samples(X):
    eccentricity = Eccentricity()
    Xt = eccentricity.fit_transform(X)
    assert Xt.shape == (len(X),)


@given(X=arrays(dtype=np.float, shape=array_shapes(min_dims=2, max_dims=2)))
def test_eccentricity_values_with_infinity_norm_equals_max_row_values(X):
    eccentricity = Eccentricity()
    Xt = eccentricity.fit_transform(X)
    distance_matrix = squareform(pdist(X))
    assert_almost_equal(Xt, np.max(distance_matrix, axis=1))
