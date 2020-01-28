import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, integers, booleans, composite
from numpy.testing import assert_almost_equal
from functools import reduce
import pytest

from sklearn.exceptions import NotFittedError
from gtda.mapper.cluster import FirstHistogramGap, FirstSimpleGap


@composite
def get_one_cluster(draw, n_points, dim):
    f = draw(arrays(dtype=np.float,
                    elements=floats(allow_nan=False,
                                    allow_infinity=False,
                                    min_value=-0.4,
                                    max_value=0.4),
                    shape=(n_points, dim), unique=False))
    return f


@composite
def get_clusters(draw, n_clusters, n_points_per_cluster, dim):
    positions = draw(arrays(dtype=np.float,
                            elements=integers(min_value=-100,
                                              max_value=100),
                            shape=(n_clusters, dim), unique=True))
    positions = np.repeat(positions, repeats=n_points_per_cluster,
                          axis=0)
    positions += draw(get_one_cluster(n_clusters * n_points_per_cluster,
                                      dim))
    return positions


@composite
def get_input(draw):
    n_clusters = draw(integers(min_value=2, max_value=5))
    n_points_per_cluster = draw(integers(min_value=2, max_value=5))
    dim = draw(integers(min_value=1, max_value=10))
    return n_points_per_cluster, n_clusters, dim, \
           draw(get_clusters(n_clusters, n_points_per_cluster, dim))


@given(inp=get_input())
def test_FirstSimpleGap(inp):
    n_points_per_cluster, n_clusters, _, pts = inp
    fs = FirstSimpleGap(relative_gap_size=0.01,
                        max_fraction=None,
                        affinity='euclidean', memory=None, linkage='single')
    preds = fs.fit_predict(pts)
    unique, counts = np.unique(preds, return_counts=True)
    # check that the nb of clusters corresponds to the nb of synth. clusters
    try:
        assert unique.shape[0] == n_clusters
    except AssertionError as e:
        print(preds)
        raise AssertionError(e)
    # check that the nb of pts in a cluster corresponds to what we expect
    assert_almost_equal(counts, n_points_per_cluster)
