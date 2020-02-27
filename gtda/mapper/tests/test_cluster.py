import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite
from numpy.testing import assert_almost_equal
from scipy.spatial import distance_matrix

from gtda.mapper import FirstHistogramGap, FirstSimpleGap


@composite
def get_one_cluster(draw, n_points, dim):
    """Get an array of n_points in a dim-dimensional space,
     in the [-1,1]-hypercube"""
    f = draw(arrays(dtype=np.float,
                    elements=floats(allow_nan=False,
                                    allow_infinity=False,
                                    min_value=-1.,
                                    max_value=1.),
                    shape=(n_points, dim), unique=False))
    return f


@composite
def get_clusters(draw, n_clusters, n_points_per_cluster, dim, std=1):
    """Get n_clusters clusters, with n_points_per_cluster points per cluster
    embedded in dim."""
    positions = np.repeat(draw(arrays(dtype=np.float,
                                      elements=integers(min_value=-100,
                                                        max_value=100),
                                      shape=(1, dim),
                                      unique=True)), repeats=n_clusters,
                          axis=0)
    positions += np.repeat(np.arange(0, n_clusters).reshape(-1, 1),
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
    return n_points_per_cluster, n_clusters, dim, draw(
        get_clusters(n_clusters, n_points_per_cluster,
                     dim, std=std))


@given(inp=get_input(n_clusters=1, n_points_per_cluster=1, std=1))
def test_on_trivial_input(inp):
    """Test that with one cluster, and one point,
    we always get one cluster, regardless of its location."""
    n_points_per_cluster, n_clusters, dim, pts = inp
    fs = FirstSimpleGap()
    fs = fs.fit(pts)
    assert fs.n_clusters_ == n_clusters

    fh = FirstHistogramGap()
    fh = fh.fit(pts)
    assert fh.n_clusters_ == n_clusters


@given(inp=get_input(std=0.02))
def test_firstsimplegap(inp):
    """For a multimodal distribution, check that the ``FirstSimpleGap``
    with appropriate parameters finds the right number of clusters,
    and that each has the correct number of points
    ``n_points_per_cluster``."""
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
    """For a multimodal distribution, check that the ``FirstHistogramGap``
    with appropriate parameters finds the right number of clusters,
    and that each has the correct number of points
    ``n_points_per_cluster``."""
    n_points_per_cluster, n_clusters, _, pts = inp
    fh = FirstHistogramGap(freq_threshold=0, max_fraction=None, n_bins_start=5,
                           affinity='euclidean', memory=None, linkage='single')
    preds = fh.fit_predict(pts)
    unique, counts = np.unique(preds, return_counts=True)
    # check that the nb of clusters corresponds to the nb of synth. clusters
    assert unique.shape[0] == n_clusters
    # check that the nb of pts in a cluster corresponds to what we expect
    assert_almost_equal(counts, n_points_per_cluster)


@given(inp=get_input(), max_frac=floats(min_value=0., exclude_min=True,
                                        max_value=1., exclude_max=True))
def test_max_fraction_clusters(inp, max_frac):
    """ Check that the clusterers (``FirstSimpleGap``,
    ``FirstHistogramGap``) respect the ``max_num_clusters`` constraint,
    if it is set."""
    n_points_per_cluster, n_clusters, _, pts = inp
    max_num_clusters = max_frac * (n_points_per_cluster * n_clusters
                                   - 1)

    fs = FirstSimpleGap(max_fraction=max_frac)
    _ = fs.fit_predict(pts)
    assert fs.n_clusters_ <= np.ceil(max_num_clusters*n_clusters)

    fh = FirstHistogramGap(max_fraction=max_frac)
    _ = fh.fit_predict(pts)
    assert fh.n_clusters_ <= np.ceil(max_num_clusters*n_clusters)


@given(inp=get_input())
def test_precomputed_distances(inp):
    """Verify that the clustering based on ``distance_matrix`` is the same
    as the clustering on points, that were used to calculate
    that distance matrix."""
    n_points_per_cluster, n_clusters, _, pts = inp

    dist_matrix = distance_matrix(pts, pts, p=2)
    fh_matrix = FirstHistogramGap(freq_threshold=0, max_fraction=None,
                                  n_bins_start=5, affinity='precomputed',
                                  memory=None, linkage='single')
    preds_mat = fh_matrix.fit_predict(dist_matrix)

    fh = FirstHistogramGap(freq_threshold=0, max_fraction=None,
                           n_bins_start=5, affinity='euclidean',
                           memory=None, linkage='single')
    preds = fh.fit_predict(pts)

    indices_cluster = set(preds)

    def get_partition_from_preds(preds):
        """From a vector of predictions (labels), get a set of frozensets,
        where each frozenset represents a cluster, and has the indices of rows
        of its elements."""
        return set([frozenset(np.where(preds == c)[0])
                    for c in indices_cluster])

    assert(get_partition_from_preds(preds)
           == get_partition_from_preds(preds_mat))
