import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
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
                                    min_value=-1.,
                                    max_value=1.),
                    shape=(n_points, dim), unique=False))
    return f


@composite
def get_clusters(draw, n_clusters, n_points_per_cluster, dim, std=1):
    positions = np.repeat(draw(arrays(dtype=np.float,
                                      elements=integers(min_value=-100,
                                                        max_value=100),
                                      shape=(1, dim),
                                      unique=True)),  repeats=n_clusters, axis=0)\
             + np.repeat(np.arange(0, n_clusters).reshape(-1, 1),
                         repeats=dim, axis=1)
    positions = np.repeat(positions, repeats=n_points_per_cluster,
                          axis=0)
    positions += std*draw(get_one_cluster(n_clusters * n_points_per_cluster,
                                          dim))
    return positions


@composite
def get_input(draw, n_clusters=None, n_points_per_cluster=None,
              dim=None, std=0.02):
    if n_clusters is None:
        n_clusters = draw(integers(min_value=2, max_value=5))
    if n_points_per_cluster is None:
        n_points_per_cluster = draw(integers(min_value=2, max_value=5))
    if dim is None:
        dim = draw(integers(min_value=1, max_value=10))
    return n_points_per_cluster, n_clusters, dim, \
           draw(get_clusters(n_clusters, n_points_per_cluster,
                             dim, std=std))


@given(inp=get_input(n_clusters=1, n_points_per_cluster=1, std=1))
def test_on_trivial_input(inp):
    n_points_per_cluster, n_clusters, dim, pts = inp
    fs = FirstSimpleGap()
    fs = fs.fit(pts)
    assert fs.n_clusters_ == n_clusters

    fh = FirstHistogramGap()
    fh = fh.fit(pts)
    assert fh.n_clusters_ == n_clusters


@given(inp=get_input(std=0.02))
def test_firstsimplegap(inp):
    n_points_per_cluster, n_clusters, _, pts = inp
    fs = FirstSimpleGap(relative_gap_size=0.5,
                        max_fraction=None,
                        affinity='euclidean', memory=None, linkage='single')
    preds = fs.fit_predict(pts).astype(int)
    unique, counts = np.unique(preds, return_counts=True)
    # check that the nb of clusters corresponds to the nb of synth. clusters
    assert unique.shape[0] == n_clusters
    # check that the nb of pts in a cluster corresponds to what we expect
    assert_almost_equal(counts, n_points_per_cluster)


@given(inp=get_input(n_clusters=2, std=0.02))
def test_firsthistogramgap(inp):
    n_points_per_cluster, n_clusters, _, pts = inp
    fh = FirstHistogramGap(freq_threshold=0, max_fraction=None, n_bins_start=5,
                           affinity='euclidean', memory=None, linkage='single')
    preds = fh.fit_predict(pts)
    unique, counts = np.unique(preds, return_counts=True)
    # check that the nb of clusters corresponds to the nb of synth. clusters
    assert unique.shape[0] == n_clusters
    # check that the nb of pts in a cluster corresponds to what we expect
    assert_almost_equal(counts, n_points_per_cluster)