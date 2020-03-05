"""Clustering methods and classes for parallelised clustering."""
# License: GNU AGPLv3

from inspect import signature
from numbers import Real

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.cluster import DBSCAN

try:  # scikit-learn >= 0.22.1
    from sklearn.cluster._agglomerative import _TREE_BUILDERS, _hc_cut
except ImportError:
    from sklearn.cluster._hierarchical import _TREE_BUILDERS, _hc_cut
from sklearn.utils import check_array
from sklearn.utils.validation import check_memory

from ..utils.intervals import Interval
from ..utils.validation import validate_params
from .utils._cluster import _num_clusters_histogram, _num_clusters_simple


class ParallelClustering(BaseEstimator):
    """Employ joblib parallelism to cluster different portions of a dataset.

    An arbitrary clustering class which stores a ``labels_`` attribute in
    ``fit`` can be passed to the constructor. Examples are most classes in
    ``sklearn.cluster``. The input of :meth:`fit` is of the form ``[X_tot,
    masks]`` where ``X_tot`` is the full dataset, and ``masks`` is a
    two-dimensional boolean array, each column of which indicates the
    location of a portion of ``X_tot`` to cluster separately. Parallelism is
    achieved over the columns of ``masks``.

    Parameters
    ----------
    clusterer : object, optional, default: ``None``
        Clustering object such as derived from
        :class:`sklearn.base.ClusterMixin`. ``None`` means that the default
        :class:`sklearn.cluster.DBSCAN` is used.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    parallel_backend_prefer : ``'processes'`` | ``'threads'``, optional, \
        default: ``'threads'``
        Selects the default joblib backend. The default process-based backend
        is 'loky' and the default thread-based backend is 'threading'.

    Attributes
    ----------
    clusterers_ : tuple of object
        If `clusterer` is not ``None``, clones of `clusterer` fitted
        to the portions of the full data array specified in :meth:`fit`.
        Otherwise, clones of a default instance of
        :class:`sklearn.cluster.DBSCAN`, fitted in the same way.

    clusters_ : list of list of tuple
       Labels and indices of each cluster found in :meth:`fit`. The i-th
       entry corresponds to the i-th portion of the data; it is a list
       of triples of the form ``(i, label, indices)``, where ``label`` is a
       cluster label and ``indices`` is the array of indices of points
       belonging to cluster ``(i, label)``.

    """

    def __init__(self, clusterer=None,
                 n_jobs=None,
                 parallel_backend_prefer='threads'):
        self.clusterer = clusterer
        self.n_jobs = n_jobs
        self.parallel_backend_prefer = parallel_backend_prefer

    def _validate_clusterer(self, default=DBSCAN()):
        """Set :attr:`clusterer_` depending on the value of  `clusterer`.

        Also verify whether calculations are to be based on precomputed
        metric/affinity information or not.

        """
        if self.clusterer is not None:
            self._clusterer = self.clusterer
        else:
            self._clusterer = default
        params = [param for param in ['metric', 'affinity']
                  if param in signature(self._clusterer.__init__).parameters]
        precomputed = [(getattr(self._clusterer, param) == 'precomputed')
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

        :attr:`clusterers_` and :attr:`clusters_` are computed and stored.

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

        self.clusterers_ = Parallel(n_jobs=self.n_jobs,
                                    prefer=self.parallel_backend_prefer)(
            delayed(single_fitter)(
                X_tot, np.flatnonzero(mask),
                mask_num, sample_weight=sample_weights[mask_num])
            for mask_num, mask in enumerate(masks.T))
        self.clusters_ = [clusterer.abs_labels_ for clusterer in
                          self.clusterers_]
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
        cloned_clusterer = clone(self._clusterer)
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

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit to the data, and return the found clusters.

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
        clusters : list of list of tuple
            See :attr:`clusters_`.

        """
        self.fit(X, sample_weight=sample_weight)
        return self.clusters_

    def transform(self, X, y=None):
        """Not implemented.

        Only present so that the class is a valid step in a scikit-learn
        pipeline.

        Parameters
        ----------
        X : Ignored
            Ignored.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        """
        raise NotImplementedError(
            "Transforming new data with a fitted ParallelClustering object "
            "not yet implemented, use fit_transform instead.")

    def fit_transform(self, X, y=None, **fit_params):
        """Alias for :meth:`fit_predict`.

        Allows for this class to be used as a step in a scikit-learn pipeline.

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

        Returns
        -------
        Xt : list of list of tuple
            See :attr:`clusters_`.

        """
        Xt = self.fit_predict(X, y, **fit_params)
        return Xt


class Agglomerative:
    """Base class for agglomerative clustering.

    Implements scikit-learn's tree building algorithms for linkage-based
    clustering. Inheriting classes may implement stopping rules for determining
    the number of clusters.

    Attributes
    ----------
    children_ : ndarray of shape (n_nodes - 1, 2)
        The children of each non-leaf node. Values less than ``n_samples``
        correspond to leaves of the tree which are the original samples.
        A node ``i`` greater than or equal to ``n_samples`` is a non-leaf
        node and has children ``children_[i - n_samples]``. Alternatively
        at the ``i``-th iteration, ``children[i][0]`` and ``children[i][1]``
        are merged to form node ``n_samples + i``.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    distances_ : ndarray of shape (n_nodes - 1,)
        Distances between nodes in the corresponding place in
        :attr:`children_`.

    """

    def _build_tree(self, X):
        memory = check_memory(self.memory)

        if self.linkage == "ward" and self.affinity != "euclidean":
            raise ValueError(f"{self.affinity} was provided as affinity. "
                             f"Ward can only work with Euclidean distances.")
        if self.linkage not in _TREE_BUILDERS:
            raise ValueError(f"Unknown linkage type {self.linkage}. Valid "
                             f"options are {_TREE_BUILDERS.keys()}")
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
    """Agglomerative clustering cutting the dendrogram at the first instance
    of a sufficiently large gap.

    A simple threshold is determined as a fraction of the largest linkage
    value in the full dendrogram. If possible, the dendrogram is cut at the
    first occurrence of a gap, between the linkage values of successive merges,
    which exceeds this threshold. Otherwise, a single cluster is returned. The
    algorithm can be partially overridden to ensure that the final number of
    clusters does not exceed a certain threshold, by passing a parameter
    `max_fraction`.

    Parameters
    ----------
    relative_gap_size : float, optional, default: ``0.3``
        The fraction of the largest linkage in the dendrogram to be used as
        a threshold for determining a large enough gap.

    max_fraction : float or None, optional, default: ``None``
        When not ``None``, the algorithm is constrained to produce no more
        than ``max_fraction * (n_samples - 1)`` clusters, even if a
        candidate gap is observed in the iterative process which would produce
        a greater number of clusters.

    affinity : str, optional, default: ``'euclidean'``
        Metric used to compute the linkage. Can be ``'euclidean'``, ``'l1'``,
        ``'l2'``, ``'manhattan'``, ``'cosine'``, or ``'precomputed'``.
        If linkage is ``'ward'``, only ``'euclidean'`` is accepted.
        If ``'precomputed'``, a distance matrix (instead of a similarity
        matrix) is needed as input for :meth:`fit`.

    memory : None, str or object with the joblib.Memory interface, \
        optional, default: ``None``
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    linkage : ``'ward'`` | ``'complete'`` | ``'average'`` | ``'single'``, \
        optional, default: ``'single'``
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

    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample.

    children_ : ndarray of shape (n_nodes - 1, 2)
        The children of each non-leaf node. Values less than ``n_samples``
        correspond to leaves of the tree which are the original samples.
        A node ``i`` greater than or equal to ``n_samples`` is a non-leaf
        node and has children ``children_[i - n_samples]``. Alternatively
        at the ``i``-th iteration, ``children[i][0]`` and ``children[i][1]``
        are merged to form node ``n_samples + i``.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    distances_ : ndarray of shape (n_nodes - 1,)
        Distances between nodes in the corresponding place in
        :attr:`children_`.

    See also
    --------
    FirstHistogramGap

    """

    _hyperparameters = {
        'relative_gap_size': {
            'type': Real, 'in': Interval(0, 1, closed='right')},
        'max_fraction': {
            'type': (Real, type(None)), 'in': Interval(0, 1, closed='right')},
        'affinity': {'type': str},
        'linkage': {'type': str}
    }

    def __init__(self, relative_gap_size=0.3, max_fraction=None,
                 affinity='euclidean', memory=None, linkage='single'):
        self.relative_gap_size = relative_gap_size
        self.max_fraction = max_fraction
        self.affinity = affinity
        self.memory = memory
        self.linkage = linkage

    def fit(self, X, y=None):
        """Fit the agglomerative clustering from features or distance matrix.

        The stopping rule is used to determine :attr:`n_clusters_`, and the
        full dendrogram is cut there to compute :attr:`labels_`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        X = check_array(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['memory'])

        _max_fraction = 1. if self.max_fraction is None else self.max_fraction

        if X.shape[0] == 1:
            self.labels_ = np.array([0])
            self.n_clusters_ = 1
            return self

        self._build_tree(X)

        min_gap_size = self.relative_gap_size * self.distances_[-1]
        self.n_clusters_ = _num_clusters_simple(
            self.distances_, min_gap_size, _max_fraction)

        # Cut the tree to find labels
        # TODO: Verify whether Daniel Mullner's implementation of this step
        #  offers any advantage
        self.labels_ = _hc_cut(self.n_clusters_, self.children_,
                               self.n_leaves_)
        return self


class FirstHistogramGap(ClusterMixin, BaseEstimator, Agglomerative):
    """Agglomerative clustering with stopping rule given by a histogram-based
    version of the first gap method, introduced in [1]_.

    Given a frequency threshold f and an initial integer k: 1) create a
    histogram of k equally spaced bins of the number of merges in the
    dendrogram, as a function of the linkage parameter; 2) the value of
    linkage at which the tree is to be cut is the first one after which a
    bin of height no greater than f (i.e. a "gap") is observed; 3) if no gap is
    observed, increase k and repeat 1) and 2) until termination. The algorithm
    can be partially overridden to ensure that the final number of clusters
    does not exceed a certain threshold, by passing a parameter `max_fraction`.

    Parameters
    ----------
    freq_threshold : int, optional, default: ``0``
        The frequency threshold for declaring that a gap in the histogram of
        merges is present.

    max_fraction : float or None, optional, default: ``None``
        When not ``None``, the algorithm is constrained to produce no more
        than ``max_fraction * (n_samples - 1)`` clusters, even if a
        candidate gap is observed in the iterative process which would produce
        a greater number of clusters.

    n_bins_start : int, optional, default: ``5``
        The initial number of bins in the iterative process for finding a
        gap in the histogram of merges.

    affinity : str, optional, default: ``'euclidean'``
        Metric used to compute the linkage. Can be ``'euclidean'``, ``'l1'``,
        ``'l2'``, ``'manhattan'``, ``'cosine'``, or ``'precomputed'``.
        If linkage is ``'ward'``, only ``'euclidean'`` is accepted.
        If ``'precomputed'``, a distance matrix (instead of a similarity
        matrix) is needed as input for :meth:`fit`.

    memory : None, str or object with the joblib.Memory interface, \
        optional, default: ``None``
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    linkage : ``'ward'`` | ``'complete'`` | ``'average'`` | ``'single'``, \
        optional, default: ``'single'``
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

    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample.

    children_ : ndarray of shape (n_nodes - 1, 2)
        The children of each non-leaf node. Values less than ``n_samples``
        correspond to leaves of the tree which are the original samples.
        A node ``i`` greater than or equal to ``n_samples`` is a non-leaf
        node and has children ``children_[i - n_samples]``. Alternatively
        at the ``i``-th iteration, ``children[i][0]`` and ``children[i][1]``
        are merged to form node ``n_samples + i``.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    distances_ : ndarray of shape (n_nodes - 1,)
        Distances between nodes in the corresponding place in
        :attr:`children_`.

    See also
    --------
    FirstSimpleGap

    References
    ----------
    .. [1] G. Singh, F. Mémoli, and G. Carlsson, "Topological methods for the
           analysis of high dimensional data sets and 3D object recognition";
           in *SPBG*, pp. 91--100, 2007.

    """

    _hyperparameters = {
        'freq_threshold': {
            'type': int, 'in': Interval(0, np.inf, closed='left')},
        'max_fraction': {
            'type': (Real, type(None)), 'in': Interval(0, 1, closed='right')},
        'n_bins_start': {
            'type': int, 'in': Interval(1, np.inf, closed='left')},
        'affinity': {'type': str},
        'linkage': {'type': str}
    }

    def __init__(self, freq_threshold=0, max_fraction=None, n_bins_start=5,
                 affinity='euclidean', memory=None, linkage='single'):
        self.freq_threshold = freq_threshold
        self.max_fraction = max_fraction
        self.n_bins_start = n_bins_start
        self.affinity = affinity
        self.memory = memory
        self.linkage = linkage

    def fit(self, X, y=None):
        """Fit the agglomerative clustering from features or distance matrix.

        The stopping rule is used to determine :attr:`n_clusters_`, and the
        full dendrogram is cut there to compute :attr:`labels_`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        X = check_array(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['memory'])

        _max_fraction = 1. if self.max_fraction is None else self.max_fraction

        if X.shape[0] == 1:
            self.labels_ = np.array([0])
            self.n_clusters_ = 1
            return self

        self._build_tree(X)

        self.n_clusters_ = _num_clusters_histogram(
            self.distances_, self.freq_threshold, self.n_bins_start,
            _max_fraction)

        # Cut the tree to find labels
        # TODO: Verify whether Daniel Mullner's implementation of this step
        #  offers any advantage
        self.labels_ = _hc_cut(self.n_clusters_, self.children_,
                               self.n_leaves_)
        return self
