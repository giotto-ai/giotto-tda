"""Testing for FirstHistogramGap and FirstSimpleGap clusterers, and testing
for ParallelClustering."""
# License: GNU AGPLv3

from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pytest
import sklearn as sk
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite
from numpy.testing import assert_almost_equal
from scipy.spatial import distance_matrix

from gtda.mapper import ParallelClustering, FirstHistogramGap, \
    FirstSimpleGap, make_mapper_pipeline


def test_parallel_clustering_bad_input():
    pc = ParallelClustering(sk.cluster.DBSCAN())
    X = [np.random.random((5, 4)), np.random.random((5, 4))]

    with pytest.raises(TypeError, match="`masks` must be a boolean array."):
        pc.fit(X)

    X[1] = np.ones((6, 4), dtype=bool)
    with pytest.raises(ValueError,
                       match="`X_tot` and `masks` must have the same number"):
        pc.fit(X)


def test_parallel_clustering_bad_clusterer():
    pc = ParallelClustering(sk.decomposition.PCA())
    X = [np.random.random((5, 4)), np.ones((5, 4), dtype=bool)]

    with pytest.raises(TypeError, match="`clusterer` must be an instance of"):
        pc.fit(X)


def test_parallel_clustering_transform_not_implemented():
    pc = ParallelClustering(sk.cluster.DBSCAN())
    X = [np.random.random((5, 4)), np.ones((5, 4), dtype=bool)]

    with pytest.raises(NotImplementedError):
        pc.transform(X)


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
@pytest.mark.parametrize("sample_weight", [None, np.random.random(5)])
def test_parallel_clustering_kmeans(n_jobs, sample_weight):
    kmeans = sk.cluster.KMeans(n_clusters=2, random_state=0)
    pc = ParallelClustering(kmeans)
    X = [np.random.random((5, 4)), np.ones((5, 4), dtype=bool)]
    single_labels = kmeans.fit_predict(X[0], sample_weight=sample_weight)
    _, inverse = np.unique(single_labels, return_inverse=True)

    res = pc.fit_predict(X, sample_weight=sample_weight)
    exp = np.empty(5, dtype=object)
    exp[:] = [tuple([])] * 5
    for i in range(4):
        labels_i = np.empty(len(single_labels), dtype=object)
        labels_i[:] = [((i, rel_label),) for rel_label in inverse]
        exp[:] += labels_i

    assert np.array_equal(res, exp)


def test_parallel_clustering_metric_affinity_precomputed_not_implemented():
    class DummyClusterer(sk.base.BaseEstimator, sk.base.ClusterMixin):
        def __init__(self, metric="precomputed", affinity="precomputed"):
            self.metric = metric
            self.affinity = affinity

    pc = ParallelClustering(DummyClusterer())
    X = [np.random.random((5, 4)), np.ones((5, 4), dtype=bool)]

    with pytest.raises(NotImplementedError,
                       match="Behaviour when metric and affinity"):
        pc.fit(X)


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_parallel_clustering_precomputed(n_jobs):
    pc = ParallelClustering(sk.cluster.DBSCAN())
    masks = np.random.choice([True, False], size=20).reshape((10, 2))
    X = [np.random.random((10, 4)), masks]
    pc_precomp = ParallelClustering(sk.cluster.DBSCAN(metric="precomputed"))
    X_precomp = [sk.metrics.pairwise_distances(X[0]), masks]

    res = pc.fit_predict(X)
    res_precomp = pc_precomp.fit_predict(X_precomp)

    assert np.array_equal(res, res_precomp)


@composite
def get_one_cluster(draw, n_points, dim):
    """Get an array of n_points in a dim-dimensional space, in the
    [-1, 1]-hypercube."""
    return draw(arrays(dtype=float,
                       elements=floats(allow_nan=False,
                                       allow_infinity=False,
                                       min_value=-1.,
                                       max_value=1.),
                       shape=(n_points, dim), unique=False))


@composite
def get_clusters(draw, n_clusters, n_points_per_cluster, dim, std=1):
    """Get n_clusters clusters, with n_points_per_cluster points per cluster
    embedded in dim."""
    positions = np.repeat(draw(arrays(dtype=float,
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


@settings(deadline=500)
@given(inp=get_input(n_clusters=1, n_points_per_cluster=1, std=1))
def test_on_trivial_input(inp):
    """Test that with one cluster, and one point, we always get one cluster,
    regardless of its location."""
    n_points_per_cluster, n_clusters, dim, pts = inp
    fs = FirstSimpleGap()
    fs = fs.fit(pts)
    assert fs.n_clusters_ == n_clusters

    fh = FirstHistogramGap()
    fh = fh.fit(pts)
    assert fh.n_clusters_ == n_clusters


@settings(deadline=500)
@given(inp=get_input(std=0.02))
def test_firstsimplegap(inp):
    """For a multimodal distribution, check that ``FirstSimpleGap`` with
    appropriate parameters finds the right number of clusters, and that each
    has the correct number of points ``n_points_per_cluster``."""
    n_points_per_cluster, n_clusters, _, pts = inp
    fs = FirstSimpleGap(relative_gap_size=0.5,
                        max_fraction=1.,
                        affinity='euclidean', memory=None, linkage='single')
    preds = fs.fit_predict(pts).astype(int)
    unique, counts = np.unique(preds, return_counts=True)
    # check that the nb of clusters corresponds to the nb of synth. clusters
    assert unique.shape[0] == n_clusters
    # check that the nb of pts in a cluster corresponds to what we expect
    assert_almost_equal(counts, n_points_per_cluster)


@settings(deadline=500)
@given(inp=get_input(n_clusters=2, std=0.02))
def test_firsthistogramgap(inp):
    """For a multimodal distribution, check that the ``FirstHistogramGap`` with
    appropriate parameters finds the right number of clusters, and that each
    has the correct number of points ``n_points_per_cluster``."""
    n_points_per_cluster, n_clusters, _, pts = inp
    fh = FirstHistogramGap(freq_threshold=0, max_fraction=1., n_bins_start=5,
                           affinity='euclidean', memory=None, linkage='single')
    preds = fh.fit_predict(pts)
    unique, counts = np.unique(preds, return_counts=True)
    # check that the nb of clusters corresponds to the nb of synth. clusters
    assert unique.shape[0] == n_clusters
    # check that the nb of pts in a cluster corresponds to what we expect
    assert_almost_equal(counts, n_points_per_cluster)


@settings(deadline=500)
@given(inp=get_input(), max_frac=floats(min_value=0., exclude_min=True,
                                        max_value=1., exclude_max=False))
def test_max_fraction_clusters(inp, max_frac):
    """ Check that ``FirstSimpleGap`` and ``FirstHistogramGap`` respect the
    ``max_num_clusters`` constraint, if it is set."""
    n_points_per_cluster, n_clusters, _, pts = inp
    max_num_clusters = max_frac * n_points_per_cluster * n_clusters

    fs = FirstSimpleGap(max_fraction=max_frac)
    _ = fs.fit_predict(pts)
    assert fs.n_clusters_ <= np.floor(max_num_clusters)

    fh = FirstHistogramGap(max_fraction=max_frac)
    _ = fh.fit_predict(pts)
    assert fh.n_clusters_ <= np.floor(max_num_clusters)


@settings(deadline=500)
@given(inp=get_input())
def test_precomputed_distances(inp):
    """Verify that the clustering based on a distance matrix is the same as
    the clustering on points used to calculate that distance matrix."""
    n_points_per_cluster, n_clusters, _, pts = inp

    dist_matrix = distance_matrix(pts, pts, p=2)
    fh_matrix = FirstHistogramGap(freq_threshold=0, max_fraction=1.,
                                  n_bins_start=5, affinity='precomputed',
                                  memory=None, linkage='single')
    preds_mat = fh_matrix.fit_predict(dist_matrix)

    fh = FirstHistogramGap(freq_threshold=0, max_fraction=1.,
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

    assert get_partition_from_preds(preds) == \
           get_partition_from_preds(preds_mat)


def test_mapper_pipeline_picklable():
    # Regression test for issue #596
    X = np.random.random((100, 2))
    cachedir = mkdtemp()
    pipe = make_mapper_pipeline(memory=cachedir)
    pipe.fit_transform(X)
    rmtree(cachedir)
