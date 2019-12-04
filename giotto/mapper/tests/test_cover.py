import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays

from giotto.mapper.cover import OneDimensionalCover


@given(X=arrays(dtype=np.float, shape=array_shapes(min_dims=1, max_dims=1)))
def test_one_dimensional_cover_shape(X):
    cover = OneDimensionalCover()
    n_samples, n_intervals = len(X), cover.n_intervals
    Xt = cover.fit_transform(X.reshape(-1, 1))
    assert (n_samples, n_intervals) == Xt.shape
