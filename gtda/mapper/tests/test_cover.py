import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, integers, booleans, composite
from numpy.testing import assert_almost_equal
from functools import reduce
import pytest

from sklearn.exceptions import NotFittedError
from gtda.mapper import OneDimensionalCover, CubicalCover


@composite
def get_filter(draw, shape=None):
    """Generate a 1d array of floats, of a given shape.
    If the shape is not given, generate a shape of at least (4,)."""
    if shape is None:
        shape = array_shapes(min_dims=1, max_dims=1,
                             min_side=4)
    points = draw(arrays(dtype=np.float,
                         elements=floats(allow_nan=False,
                                         allow_infinity=False,
                                         min_value=-1e10,
                                         max_value=1e10),
                         shape=shape, unique=True))
    return points


@composite
def get_nb_intervals(draw):
    nb_intervals = draw(integers(min_value=3, max_value=20))
    return nb_intervals


@composite
def get_overlap_fraction(draw):
    overlap = draw(floats(allow_nan=False,
                          allow_infinity=False,
                          min_value=1e-8, exclude_min=True,
                          max_value=1., exclude_max=True),
                   )
    return overlap


@composite
def get_kind(draw):
    is_uniform = draw(booleans())
    return 'uniform' if is_uniform else 'balanced'


@given(
    filter_values=get_filter(),
    n_intervals=get_nb_intervals()
)
def test_one_dimensional_cover_shape(filter_values, n_intervals):
    """Assert that the length of the mask ``unique_interval_masks``
    corresponds to the pre-specified ``n_samples`` and that there
    are no more intervals in the cover than ``n_intervals``.
    The case when the filter has only a unique value, in which
    case the fit_transform should throw an error, is treated separately.
    """
    # TODO: Extend to inputs with shape (n_samples, 1)
    cover = OneDimensionalCover(n_intervals=n_intervals)
    n_samples, n_intervals = len(filter_values), cover.n_intervals
    try:
        unique_interval_masks = cover.fit_transform(filter_values)
        assert n_samples == unique_interval_masks.shape[0]
        assert n_intervals >= unique_interval_masks.shape[1]
    except ValueError as ve:
        assert ve.args[0] == f"Only one unique filter value found, cannot " \
                             f"fit {n_intervals} > 1 intervals."
        assert (n_intervals > 1) and (len(np.unique(filter_values)) == 1)


@given(
    filter_values=get_filter()
)
def test_filter_values_covered_by_single_interval(filter_values):
    """Verify that a single intervals covers all the values in
    ``filter_values``"""
    # TODO: Extend to inputs with shape (n_samples, 1)
    cover = OneDimensionalCover(n_intervals=1)
    interval_masks = cover.fit_transform(filter_values)
    # TODO: Generate filter_values with desired shape
    assert_almost_equal(
        filter_values[:, None][interval_masks], filter_values)


@given(
    filter_values=get_filter(),
    n_intervals=get_nb_intervals(),
    overlap_frac=get_overlap_fraction()
)
def test_equal_interval_length(filter_values, n_intervals, overlap_frac):
    """Test that all the intervals have the same length, up to an additive
    constant of 0.1."""
    cover = OneDimensionalCover(kind='uniform',
                                n_intervals=n_intervals,
                                overlap_frac=overlap_frac)
    cover = cover.fit(filter_values)

    lower_limits, upper_limits = np.array(
        list(map(tuple, zip(*cover.get_fitted_intervals()[1:-1]))))

    # rounding precision
    decimals = 10
    assert len(set(np.floor((upper_limits - lower_limits) *
                            decimals).tolist())) == 1


@composite
def get_input_tests_balanced(draw):
    """Points, nb_in_each_interval and nb_intervals"""
    nb_intervals = draw(get_nb_intervals())
    nb_in_each_interval = draw(integers(min_value=2, max_value=5))
    points = draw(get_filter(shape=(nb_in_each_interval * nb_intervals,)))
    return [points, nb_in_each_interval, nb_intervals]


@given(input=get_input_tests_balanced())
def test_balanced_is_balanced(input):
    """Test that each point is in one interval, and that each interval has
    ``nb_in_each_interval`` points."""
    points, nb_in_each_interval, nb_intervals = input
    oneD_cover = OneDimensionalCover(kind='balanced',
                                     n_intervals=nb_intervals,
                                     overlap_frac=0.01)
    mask = oneD_cover.fit_transform(points)
    # each interval contains nb_in_each_interval points
    assert all([s == nb_in_each_interval for s in np.sum(mask, axis=0)])
    # each point is in exactly one interval
    assert all([s == 1 for s in np.sum(mask, axis=1)])


@given(
    filter_values=get_filter(),
    n_intervals=get_nb_intervals()
)
def test_filter_values_covered_by_interval_union(filter_values,
                                                 n_intervals):
    """Test that each value is at least in one interval.
    (that is, the cover is a true cover)."""
    # TODO: Extend to inputs with shape (n_samples, 1)
    cover = OneDimensionalCover(n_intervals=n_intervals)
    interval_masks = cover.fit_transform(filter_values)
    intervals = [filter_values[interval_masks[:, i]]
                 for i in range(interval_masks.shape[1])]
    intervals_union = reduce(np.union1d, intervals)
    filter_values_union = filter_values[np.in1d(filter_values,
                                                intervals_union)]
    assert_almost_equal(filter_values_union, filter_values)


@given(
    pts=get_filter(),
    n_intervals=get_nb_intervals(),
    overlap_frac=get_overlap_fraction(),
    kind=get_kind()
)
def test_fit_transform_against_fit_and_transform(pts, n_intervals,
                                                 kind, overlap_frac):
    """Fitting and transforming should give the same result as fit_transform"""
    cover = OneDimensionalCover(n_intervals=n_intervals, kind=kind,
                                overlap_frac=overlap_frac)
    x_fit_transf = cover.fit_transform(pts)

    cover2 = OneDimensionalCover(n_intervals=n_intervals, kind=kind,
                                 overlap_frac=overlap_frac)
    cover2 = cover2.fit(pts)
    x_fit_and_transf = cover2.transform(pts)
    assert_almost_equal(x_fit_transf, x_fit_and_transf)


def test_fit_transform_limits_not_computed():
    """We do not compute intervals when `kind`= `'balanced'`,
    unless fit is explicitly called."""
    cover = OneDimensionalCover(n_intervals=10, kind='balanced',
                                overlap_frac=0.3)
    x = np.arange(0, 30)
    _ = cover.fit_transform(x)
    with pytest.raises(NotFittedError):
        _ = cover.get_fitted_intervals()


@given(pts=get_filter(shape=array_shapes(min_dims=2, max_dims=2, min_side=2)))
def test_two_dimensional_tensor(pts):
    """Verify that the oneDimensionalCover fails for an input
    with more than one dimension, and that the CubicalCover
    does not."""
    one_d = OneDimensionalCover()
    with pytest.raises(ValueError):
        one_d.fit(pts)
    cubical = CubicalCover()
    _ = cubical.fit(pts)


@given(filter=get_filter(),
       kind=get_kind(),
       n_intervals=get_nb_intervals(),
       overlap_fraction=get_overlap_fraction(),)
def test_cubical_fit_transform_consistent_with_OneD(filter, kind,
                                                    n_intervals,
                                                    overlap_fraction):
    """Check that CubicalCover gives the same results as OneDimensionalCover,
    on one-d data """
    one_d = OneDimensionalCover(kind, n_intervals, overlap_fraction)
    cubical = CubicalCover(kind, n_intervals, overlap_fraction)
    x_one_d = one_d.fit_transform(filter)
    x_cubical = cubical.fit_transform(filter)
    assert_almost_equal(x_one_d, x_cubical)


@given(filter=get_filter(),
       kind=get_kind(),
       n_intervals=get_nb_intervals(),
       overlap_fraction=get_overlap_fraction(),
       )
def test_cubical_fit_A_transform_consistent_with_OneD(filter, kind,
                                                      n_intervals,
                                                      overlap_fraction):
    """Check that CubicalCover gives the same results as OneDimensionalCover,
    on one-d data """
    one_d = OneDimensionalCover(kind, n_intervals, overlap_fraction)
    cubical = CubicalCover(kind, n_intervals, overlap_fraction)
    x_one_d = one_d.fit(filter).transform(filter)
    x_cubical = cubical.fit(filter).transform(filter)
    assert_almost_equal(x_one_d, x_cubical)
