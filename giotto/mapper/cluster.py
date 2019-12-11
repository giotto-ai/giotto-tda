from functools import partial
from inspect import signature

import numpy as np

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_memory

from sklearn.cluster import DBSCAN
from sklearn.cluster._hierarchical import _TREE_BUILDERS, _hc_cut


class ParallelClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, clusterer=None, n_jobs_outer=None, prefer='threads'):
        self.clusterer = clusterer
        self.n_jobs_outer = n_jobs_outer
        self.prefer = prefer

    def _validate_clusterer(self, default=DBSCAN()):
        """Depending on the value of parameter `clusterer`, set
        :attr:`clusterer_`. Also verify whether calculations are to be based
        on precomputed metric/affinity information or not.

        """
        if self.clusterer is not None:
            self.clusterer_ = self.clusterer
        else:
            self.clusterer_ = default
        params = [param for param in ['metric', 'affinity']
                  if param in signature(self.clusterer_.__init__).parameters]
        precomputed = [(getattr(self.clusterer_, param) == 'precomputed')
                       for param in params]
        if not precomputed:
            self._precomputed = False
        elif len(precomputed) == 1:
            self._precomputed = precomputed[0]
        else:
            raise NotImplementedError("Behaviour when metric and affinity "
                                      "are both set to 'precomputed' not yet"
                                      "implemented by ParallelClustering.")

    def fit(self, X, y=None, sample_weight=None):
        self._validate_clusterer()
        X_orig, masks = X
        if sample_weight is not None:
            sample_weights = [sample_weight[masks[:, i]]
                              for i in range(masks.shape[1])]
        else:
            sample_weights = [None] * masks.shape[1]

        if self._precomputed:
            single_fitter = self._fit_single_abs_labels_precomputed
        else:
            single_fitter = self._fit_single_abs_labels

        self.clusterers_ = Parallel(
            n_jobs=self.n_jobs_outer, prefer=self.prefer)(
            delayed(single_fitter)(
                X_orig, np.flatnonzero(mask),
                mask_num, sample_weight=sample_weights[mask_num])
            for mask_num, mask in enumerate(masks.T))
        return self

    def _fit_single_abs_labels(self, X, relative_indices, mask_num,
                               sample_weight=None):
        cloned_clusterer, unique_labels, unique_labels_inverse = \
            self._fit_single(X, relative_indices, sample_weight)
        self._create_abs_labels(cloned_clusterer, relative_indices, mask_num,
                                unique_labels, unique_labels_inverse)
        return cloned_clusterer

    def _fit_single_abs_labels_precomputed(self, X, relative_indices, mask_num,
                                           sample_weight=None):
        relative_2d_indices = np.ix_(relative_indices, relative_indices)
        cloned_clusterer, unique_labels, unique_labels_inverse = \
            self._fit_single(X, relative_2d_indices, sample_weight)
        self._create_abs_labels(cloned_clusterer, relative_indices, mask_num,
                                unique_labels, unique_labels_inverse)
        return cloned_clusterer

    def _fit_single(self, X, relative_indices, sample_weight):
        cloned_clusterer = clone(self.clusterer_)
        X_sub = X[relative_indices]

        fit_params = signature(cloned_clusterer.fit).parameters
        if 'sample_weight' in fit_params:
            cloned_clusterer.fit(X_sub, sample_weight=sample_weight)
        else:
            cloned_clusterer.fit(X_sub)

        unique_labels, unique_labels_inverse = np.unique(
            cloned_clusterer.labels_, return_inverse=True)
        return cloned_clusterer, unique_labels, unique_labels_inverse

    @staticmethod
    def _create_abs_labels(cloned_clusterer, relative_indices, mask_num,
                           unique_labels, inv):
        cloned_clusterer.abs_labels_ = [
            (mask_num, label, relative_indices[inv == i])
            for i, label in enumerate(unique_labels)]

    def transform(self, X, y=None, sample_weight=None):
        # TODO consider whether this is better implemented using decorators
        check_is_fitted(self, ['clusterers_'])
        Xt = [clusterer.abs_labels_ for clusterer in self.clusterers_]
        return Xt

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        Xt = [clusterer.abs_labels_ for clusterer in self.clusterers_]
        return Xt


class Agglomerative:
    def _build_tree(self, X):
        memory = check_memory(self.memory)

        if self.linkage == "ward" and self.affinity != "euclidean":
            raise ValueError("{} was provided as affinity. Ward can only work"
                             "with Euclidean distances.".format(self.affinity))
        if self.linkage not in _TREE_BUILDERS:
            raise ValueError("Unknown linkage type {}. Valid options are {}"
                             .format(self.linkage, _TREE_BUILDERS.keys()))
        tree_builder = _TREE_BUILDERS[self.linkage]

        # Construct the tree
        kwargs = {}
        if self.linkage != 'ward':
            kwargs['linkage'] = self.linkage
            kwargs['affinity'] = self.affinity

        out = memory.cache(tree_builder)(
            X, n_clusters=None, return_distance=True, **kwargs)

        # Scikit-learn's tree_builder returns a tuple (children,
        # n_connected_components, n_leaves, parent, distances)
        self.children_, _, self.n_leaves_, _, self.distances_ = out


class FirstSimpleGap(ClusterMixin, BaseEstimator, Agglomerative):
    """First gap method used to determine where to cut the full dendrogram
    produced by a linkage algorithm.

    """
    def __init__(self, relative_gap_size=0.3, affinity='euclidean',
                 memory=None, linkage='single'):
        self.relative_gap_size = relative_gap_size
        self.affinity = affinity
        self.memory = memory
        self.linkage = linkage

    def fit(self, X, y=None):
        """Fit the hierarchical clustering from features or distance matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        X = check_array(X)
        if X.shape[0] == 1:
            self.labels_ = np.array([0])
            return self

        self._build_tree(X)

        min_gap_size = self.relative_gap_size * self.distances_[-1]
        self.n_clusters_ = _num_clusters_simple(self.distances_, min_gap_size)

        # Cut the tree to find labels
        # TODO verify whether Daniel Mullner's implementation of this step
        #  is really needed
        self.labels_ = _hc_cut(self.n_clusters_, self.children_,
                               self.n_leaves_)
        return self


class FirstHistogramGap(ClusterMixin, BaseEstimator, Agglomerative):
    """First gap method used to determine where to cut the full dendrogram
    produced by a linkage algorithm.

    """
    def __init__(self, freq_threshold=0, n_bins_start=5, affinity='euclidean',
                 memory=None, linkage='single'):
        self.freq_threshold = freq_threshold
        self.n_bins_start = n_bins_start
        self.affinity = affinity
        self.memory = memory
        self.linkage = linkage

    def fit(self, X, y=None):
        """Fit the hierarchical clustering from features or distance matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        X = check_array(X)
        if X.shape[0] == 1:
            self.labels_ = np.array([0])
            return self

        self._build_tree(X)

        self.n_clusters_ = _num_clusters_histogram(
            self.distances_, self.freq_threshold, self.n_bins_start)

        # Cut the tree to find labels
        # TODO verify whether Daniel Mullner's implementation of this step
        #  is really needed
        self.labels_ = _hc_cut(self.n_clusters_, self.children_,
                               self.n_leaves_)
        return self


def _num_clusters_simple(distances, min_gap_size):
    # Differences between subsequent elements (padding by the first
    # distance)
    diff = np.ediff1d(distances, to_begin=distances[0])
    gap_indices = np.flatnonzero(diff >= min_gap_size)
    if gap_indices.size:
        num_clust = distances.size + 1 - gap_indices[0]
        return num_clust
    # No big enough gaps -> one cluster
    return 1


def _num_clusters_histogram(distances, freq_threshold, n_bins_start):
    if distances.size == 1:
        return 1

    if not freq_threshold:
        threshold_func = _zero_bins
    else:
        threshold_func = partial(_bins_below_threshold, freq_threshold)
    zero_bins = False
    i = 0
    while not zero_bins:
        hist, edges = np.histogram(distances, bins=n_bins_start + i)
        zero_bins_indices = threshold_func(hist)
        zero_bins = zero_bins_indices.size
        i += 1
    first_gap = zero_bins_indices[0]
    left_bin_edge_first_gap = edges[first_gap]
    gap_idx = (distances <= left_bin_edge_first_gap).sum()
    num_clust = distances.size + 1 - gap_idx
    return num_clust


def _zero_bins(hist):
    return np.flatnonzero(~hist.astype(bool))


def _bins_below_threshold(freq_threshold, hist):
    return np.flatnonzero(hist < freq_threshold)


