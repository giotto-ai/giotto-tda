import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, integers
from numpy.testing import assert_almost_equal

from gtda.mapper.cover import OneDimensionalCover


@given(
    filter_values=arrays(
        dtype=np.float,
        elements=floats(min_value=-1e100, allow_nan=False,
                        allow_infinity=False),
        shape=integers(min_value=2, max_value=1e3)),
    n_intervals=integers(min_value=1, max_value=100)
)
def test_one_dimensional_cover_shape(filter_values, n_intervals):
    # TODO: Extend to inputs with shape (n_samples, 1)
    cover = OneDimensionalCover(n_intervals=n_intervals)
    n_samples, n_intervals = len(filter_values), cover.n_intervals
    try:
        unique_interval_masks = cover.fit_transform(filter_values)
        assert n_samples == unique_interval_masks.shape[0]
        assert n_intervals >= unique_interval_masks.shape[1]
    except ValueError as ve:
        assert ve.args[0] == "Only one unique filter value found, cannot " \
                             "fit {} > 1 intervals.".format(n_intervals)
        assert (n_intervals > 1) and (len(np.unique(filter_values)) == 1)


@given(
    filter_values=arrays(dtype=np.float,
                         elements=floats(allow_nan=False,
                                         allow_infinity=False,
                                         max_value=1e3),
                         shape=integers(min_value=1, max_value=100)
                         )
)
def test_filter_values_covered_by_single_interval(filter_values):
    # TODO: Extend to inputs with shape (n_samples, 1)
    cover = OneDimensionalCover(n_intervals=1)
    interval_masks = cover.fit_transform(filter_values)
    # TODO: Generate filter_values with desired shape
    assert_almost_equal(
        filter_values[:, None][interval_masks], filter_values)


# @given(
#     filter_values=arrays(dtype=np.float,
#                          elements=floats(allow_nan=False,
#                                          allow_infinity=False,
#                                          max_value=1e3),
#                          shape=array_shapes(min_dims=1, max_dims=1,
#                                             min_side=2),
#                          unique=True),
#     n_intervals=integers(min_value=1, max_value=10)
# )
# def test_filter_values_covered_by_interval_union(filter_values, n_intervals):
#     # TODO: Extend to inputs with shape (n_samples, 1)
#     cover = OneDimensionalCover(n_intervals=n_intervals)
#     interval_masks = cover.fit_transform(filter_values)
#     intervals = [filter_values[interval_masks[:, i]]
#                  for i in range(interval_masks.shape[1])]
#     intervals_union = reduce(np.union1d, intervals)
#     filter_values_union = filter_values[np.in1d(
#         filter_values, intervals_union)]
#     assert_almost_equal(filter_values_union, filter_values)


@given(
    filter_values=arrays(dtype=np.float,
                         elements=floats(allow_nan=False,
                                         allow_infinity=False,
                                         min_value=-1e10,
                                         max_value=1e10),
                         shape=array_shapes(min_dims=1, max_dims=1,
                                            min_side=2),
                         unique=True),
    n_intervals=integers(min_value=3, max_value=50),
    overlap_frac=floats(allow_nan=False,
                        allow_infinity=False,
                        min_value=0,
                        max_value=1)
)
def test_equal_interval_length(filter_values, n_intervals, overlap_frac):
    cover = OneDimensionalCover(kind='uniform',
                                n_intervals=n_intervals,
                                overlap_frac=overlap_frac)
    cover = cover.fit(filter_values)

    lower_limits,\
        upper_limits = np.array(list(map(tuple,
                                         zip(*cover.fitted_intervals()[1:-1]))
                                     ))

    # rounding precision
    decimals = 10
    assert len(set(np.floor((upper_limits - lower_limits) *
                            decimals).tolist())) == 1


# @given(
#     filter_values=arrays(dtype=np.float,
#                          elements=floats(allow_nan=False,
#                                          allow_infinity=False,
#                                          min_value=-1e10,
#                                          max_value=1e10),
#                          shape=array_shapes(min_dims=1, max_dims=1,
#                                             min_side=2),
#                          unique=True),
#     n_intervals=integers(min_value=4, max_value=50),
#     overlap_frac=floats(allow_nan=False,
#                         allow_infinity=False,
#                         min_value=0.01,
#                         max_value=1.)
# )
# def test_overlap_fraction(filter_values, n_intervals, overlap_frac):
#     cover = OneDimensionalCover(kind='uniform',
#                                 n_intervals=n_intervals,
#                                 overlap_frac=overlap_frac)
#     cover.fit(filter_values)

#     lower_limits = cover.left_limits_[1:-1]
#     upper_limits = cover.right_limits_[1:-1]
#     lengths = (upper_limits[:-1] - lower_limits[:-1])

#     overlap_array = (upper_limits[:-1] - lower_limits[1:]) / lengths
#     expected_overlap_array = np.array([overlap_frac] * (n_intervals - 3))

#     assert_allclose(overlap_array, expected_overlap_array, rtol=1e-5)
