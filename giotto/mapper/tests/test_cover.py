import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers
from numpy.testing import assert_almost_equal
from functools import reduce

from giotto.mapper.cover import OneDimensionalCover


@given(
    filter_values=arrays(
        dtype=np.float,
        elements=floats(allow_nan=False, allow_infinity=False),
        shape=integers(min_value=2, max_value=1e3)),
    n_intervals=integers(min_value=1, max_value=100)
)
def test_one_dimensional_cover_shape(filter_values, n_intervals):
    cover = OneDimensionalCover(n_intervals=n_intervals)
    n_samples, n_intervals = len(filter_values), cover.n_intervals
    interval_masks = cover.fit_transform(filter_values)
    assert (n_samples, n_intervals) == interval_masks.shape


@given(
    filter_values=arrays(dtype=np.float,
                         elements=floats(allow_nan=False,
                                         allow_infinity=False,
                                         max_value=1e3),
                         shape=integers(min_value=1, max_value=100)
                         )
)
def test_filter_values_covered_by_single_interval(filter_values):
    cover = OneDimensionalCover(n_intervals=1)
    interval_masks = cover.fit_transform(filter_values)
    # TODO: generate filter_values with desired shape
    assert_almost_equal(
        filter_values[:, None][interval_masks], filter_values)


@given(
    filter_values=arrays(dtype=np.float,
                         elements=floats(allow_nan=False,
                                         allow_infinity=False,
                                         max_value=1e3),
                         shape=integers(min_value=1, max_value=100),
                         unique=True),
    n_intervals=integers(min_value=1, max_value=10)
)
def test_filter_values_covered_by_interval_union(filter_values, n_intervals):
    cover = OneDimensionalCover(n_intervals=n_intervals)
    interval_masks = cover.fit_transform(filter_values)
    intervals = [filter_values[interval_masks[:, i]]
                 for i in range(interval_masks.shape[1])]
    intervals_union = reduce(np.union1d, intervals)
    filter_values_union = filter_values[np.in1d(
        filter_values, intervals_union)]
    assert_almost_equal(filter_values_union, filter_values)
