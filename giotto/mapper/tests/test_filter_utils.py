"""Testing for filter utils."""
# License : Apache 2.0

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from giotto.mapper import Eccentricity
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes


@given(X=arrays(dtype=np.float, shape=array_shapes(min_dims=2, max_dims=2)))
def test_eccentricity_transform_shape_equals_number_of_samples(X):
    ecc = Eccentricity()
    X_ecc = ecc.fit_transform(X)
    assert X_ecc.shape == (len(X),)
