from inspect import signature

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.cluster import DBSCAN
from sklearn.cluster._hierarchical import _TREE_BUILDERS, _hc_cut
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_memory

from .utils._cluster import _num_clusters_histogram, _num_clusters_simple


class ParallelClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    """Employ joblib parallelism to cluster different portions of a dataset.

    Parameters
    ----------
    clusterer : object, optional, default: ``None``
        Clustering object such as derived from
        :class:`sklearn.base.ClusterMixin`. ``None`` means that the default
        :class:`sklearn.cluster.DBSCAN` is used.

    n_jobs_outer : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    prefer : ``'processes'`` | ``'threads'``, optional, default: ``'threads'``
        Selects the default joblib backend. The default process-based backend
        is 'loky' and the default thread-based backend is 'threading'.

    Attributes
    ----------
    clusterer_ : object
        Unless `clusterer` is ``None``, same as `clusterer`. Cloned prior
        to fitting each part of the data.

    clusterers_ : tuple of object
        Clones of :attr:`clusterer_`, fitted to the portions of the full
        data array specified in :meth:`fit`.

    """
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
        """Fit the clusterer on each portion of the data.

        Parameters
        ----------
        X : list-like of form ``[X_tot, masks]``
            Input data as a list of length 2. ``X_tot`` is an ndarray of shape
            (n_samples, n_features) or (n_samples, n_samples) specifying the
            full data. ``masks`` is a boolean ndarray of shape
            (n_samples, n_portions) whose columns are boolean masks
            on ``X_tot``, specifying the portions of ``X_tot`` to be
            independently clustered.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        sample_weight : array-like or None, optional, default: ``None``
            The weights for each observation in the full data. If ``None``,
            all observations are assigned equal weight. Otherwise, it has
            shape (n_samples,).

        Returns
        -------
        self : object

        """
        self._validate_clusterer()
        X_tot, masks = X
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
                X_tot, np.flatnonzero(mask),
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
    """Base class for agglomerative clustering algorithms employing a
    stopping rule for determining the number of clusters.

    Attributes
    ----------
    children_ : ndarray, shape (n_nodes - 1, 2)
        The children of each non-leaf node. Values less than ``n_samples``
        correspond to leaves of the tree which are the original samples.
        A node ``i`` greater than or equal to ``n_samples`` is a non-leaf
        node and has children ``children_[i - n_samples]``. Alternatively
        at the ``i``-th iteration, ``children[i][0]`` and ``children[i][1]``
        are merged to form node ``n_samples + i``.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    distances_ : ndarray, shape (n_nodes - 1,)
        Distances between nodes in the corresponding place in
        :attr:`children_`.

    """
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
    """Agglomerative clustering with stopping rule given by a threshold-based
    version of the first gap method.

    A simple threshold is determined as a fraction of the largest linkage
    value in the full dendrogram. If possible, the dendrogram is cut at the
    first occurrence of a gap, between the linkage values of successive merges,
    which exceeds this threshold. Otherwise, a single cluster is returned.

    Parameters
    ----------
    relative_gap_size : float, optional, default: ``0.3``
        The fraction of the largest linkage in the dendrogram to be used as
        a threshold for determining a large enough gap.

    affinity : str, optional, default: ``'euclidean'``
        Metric used to compute the linkage. Can be ``'euclidean'``, ``'l1'``,
        ``'l2'``, ``'manhattan'``, ``'cosine'``, or ``'precomputed'``.
        If linkage is ``'ward'``, only ``'euclidean'`` is accepted.
        If ``'precomputed'``, a distance matrix (instead of a similarity
        matrix) is needed as input for :meth:`fit`.

    memory : None, str or object with the joblib.Memory interface,
        optional, default: ``None``
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    linkage : ``'ward'`` | ``'complete'`` | ``'average'`` | ``'single'``, \
        optional, default: ``'single'``)
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
        - ``'ward'`` minimizes the variance of the clusters being merged.
        - ``'average'`` uses the average of the distances of each observation
          of the two sets.
        - ``'complete'`` linkage uses the maximum distances between
          all observations of the two sets.
        - ``'single'`` uses the minimum of the distances between all
          observations of the two sets.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.

    labels_ : ndarray, shape (n_samples,)
        Cluster labels for each sample.

    children_ : ndarray, shape (n_nodes - 1, 2)
        The children of each non-leaf node. Values less than ``n_samples``
        correspond to leaves of the tree which are the original samples.
        A node ``i`` greater than or equal to ``n_samples`` is a non-leaf
        node and has children ``children_[i - n_samples]``. Alternatively
        at the ``i``-th iteration, ``children[i][0]`` and ``children[i][1]``
        are merged to form node ``n_samples + i``.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    distances_ : ndarray, shape (n_nodes - 1,)
        Distances between nodes in the corresponding place in
        :attr:`children_`.

    See also
    --------
    FirstHistogramGap

    """
    def __init__(self, relative_gap_size=0.3, affinity='euclidean',
                 memory=None, linkage='single'):
        self.relative_gap_size = relative_gap_size
        self.affinity = affinity
        self.memory = memory
        self.linkage = linkage

    def fit(self, X, y=None):
        """Fit the agglomerative clustering from features or distance matrix.

        The stopping rule is used to determine :attr:`n_clusters_``, and the
        full dendrogram is cut there to compute :attr:`labels_`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : ignored
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
        #  offers any advantage
        self.labels_ = _hc_cut(self.n_clusters_, self.children_,
                               self.n_leaves_)
        return self


class FirstHistogramGap(ClusterMixin, BaseEstimator, Agglomerative):
    """Agglomerative clustering with stopping rule given by a histogram-based
    version of the first gap method.

    Given a frequency threshold f and an initial integer k: 1) create a
    histogram of k equally spaced bins of the number of merges in the
    dendrogram, as a function of the linkage parameter; 2) the value of
    linkage at which the tree is to be cut is the first one after which a
    bin of height no greater than f (i.e. a "gap") is observed; 3) if no gap is
    observed, increase k and repeat 1) and 2) until termination.

    Parameters
    ----------
    freq_threshold : int, optional, default: ``0``
        The frequency threshold for declaring that a gap in the histogram of
        merges is present.

    n_bins_start : int, optional, default: ``5``
        The initial number of bins in the iterative process for finding a
        gap in the histogram of merges.

    affinity : str, optional, default: ``'euclidean'``
        Metric used to compute the linkage. Can be ``'euclidean'``, ``'l1'``,
        ``'l2'``, ``'manhattan'``, ``'cosine'``, or ``'precomputed'``.
        If linkage is ``'ward'``, only ``'euclidean'`` is accepted.
        If ``'precomputed'``, a distance matrix (instead of a similarity
        matrix) is needed as input for :meth:`fit`.

    memory : None, str or object with the joblib.Memory interface,
        optional, default: ``None``
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    linkage : ``'ward'`` | ``'complete'`` | ``'average'`` | ``'single'``, \
        optional, default: ``'single'``)
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
        - ``'ward'`` minimizes the variance of the clusters being merged.
        - ``'average'`` uses the average of the distances of each observation
          of the two sets.
        - ``'complete'`` linkage uses the maximum distances between
          all observations of the two sets.
        - ``'single'`` uses the minimum of the distances between all
          observations of the two sets.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm.

    labels_ : ndarray, shape (n_samples,)
        Cluster labels for each sample.

    children_ : ndarray, shape (n_nodes - 1, 2)
        The children of each non-leaf node. Values less than ``n_samples``
        correspond to leaves of the tree which are the original samples.
        A node ``i`` greater than or equal to ``n_samples`` is a non-leaf
        node and has children ``children_[i - n_samples]``. Alternatively
        at the ``i``-th iteration, ``children[i][0]`` and ``children[i][1]``
        are merged to form node ``n_samples + i``.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    distances_ : ndarray, shape (n_nodes - 1,)
        Distances between nodes in the corresponding place in
        :attr:`children_`.

    See also
    --------
    FirstSimpleGap

    """
    def __init__(self, freq_threshold=0, n_bins_start=5, affinity='euclidean',
                 memory=None, linkage='single'):
        self.freq_threshold = freq_threshold
        self.n_bins_start = n_bins_start
        self.affinity = affinity
        self.memory = memory
        self.linkage = linkage

    def fit(self, X, y=None):
        """Fit the agglomerative clustering from features or distance matrix.

        The stopping rule is used to determine :attr:`n_clusters_``, and the
        full dendrogram is cut there to compute :attr:`labels_`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : ignored
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
        #  offers any advantage
        self.labels_ = _hc_cut(self.n_clusters_, self.children_,
                               self.n_leaves_)
        return self
