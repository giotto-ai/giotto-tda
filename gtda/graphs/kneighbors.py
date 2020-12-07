"""kNN graphs from point cloud data."""
# License: GNU AGPLv3

from functools import partial

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_is_fitted

from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import check_point_clouds


@adapt_fit_transform_docs
class KNeighborsGraph(BaseEstimator, TransformerMixin):
    """Adjacency matrices of :math:`k`-nearest neighbor graphs.

    Given a two-dimensional array of row vectors seen as points in
    high-dimensional space, the corresponding :math:`k`NN graph is a directed
    graph with a vertex for every vector in the array, and a directed edge from
    vertex :math:`i` to vertex :math:`j \\neq i` whenever vector :math:`j` is
    among the :math:`k` nearest neighbors of vector :math:`i`.

    Parameters
    ----------
    n_neighbors : int, optional, default: ``4``
        Number of neighbors to use. A point is not considered as its own
        neighbour.

    mode : ``'connectivity'`` | ``'distance'``, optional, \
        default: ``'connectivity'``
        Type of returned matrices: ``'connectivity'`` will return the 0-1
        connectivity matrices, and ``'distance'`` will return the distances
        between neighbors according to the given metric.

    metric : string or callable, optional, default: ``'euclidean'``
        The distance metric to use. See the documentation of
        :class:`sklearn.neighbors.DistanceMetric` for a list of available
        metrics. If set to ``'precomputed'``, input data is interpreted as a
        collection of distance matrices.

    p : int, optional, default: ``2``
        Parameter for the Minkowski (i.e. :math:`\\ell^p`) metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. Only relevant
        when `metric` is ``'minkowski'``. `p` = 1 is the Manhattan distance,
        and `p` = 2 reduces to the Euclidean distance.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.graphs import KNeighborsGraph
    >>> X = np.array([[[0, 1, 3, 0, 0],
    ...                [1, 0, 5, 0, 0],
    ...                [3, 5, 0, 4, 0],
    ...                [0, 0, 4, 0, 0]]])
    >>> kng = KNeighborsGraph(n_neighbors=2)
    >>> Xg = kng.fit_transform(X)
    >>> print(Xg[0].toarray())
    [[0. 1. 0. 1.]
     [1. 0. 0. 1.]
     [1. 0. 0. 1.]
     [1. 1. 0. 0.]]

    See also
    --------
    TransitionGraph, GraphGeodesicDistance

    Notes
    -----
    :func:`sklearn.neighbors.kneighbors_graph` is used to compute the
    adjacency matrices of kNN graphs.

    """

    def __init__(self, n_neighbors=4, mode='connectivity', metric='euclidean',
                 p=2, metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_points, n_dimensions) or (n_samples, n_points, n_points)
            Input data representing a collection of point clouds. Each entry
            in `X` is a 2D array of shape ``(n_points, n_dimensions)`` if
            `metric` is not ``'precomputed'``, or a 2D array or sparse matrix
            of shape ``(n_points, n_points)`` if `metric` is ``'precomputed'``.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        self._is_precomputed = self.metric == 'precomputed'
        check_point_clouds(X, accept_sparse=True,
                           distance_matrices=self._is_precomputed)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute kNN graphs and return their adjacency matrices in sparse
        format.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_points, n_dimensions) or (n_samples, n_points, n_points)
            Input data representing a collection of point clouds. Each entry
            in `X` is a 2D array of shape ``(n_points, n_dimensions)`` if
            `metric` is not ``'precomputed'``, or a 2D array or sparse matrix
            of shape ``(n_points, n_points)`` if `metric` is ``'precomputed'``.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : list of length n_samples
            Adjacency matrices of kNN graphs, in sparse CSR format. The
            matrices contain ones and zeros if `mode` is ``'connectivity'``,
            and floats representing distances according to `metric` if `mode`
            is ``'distance'``.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_point_clouds(X, accept_sparse=True,
                                distance_matrices=self._is_precomputed)

        _adjacency_matrix_func = partial(
            kneighbors_graph, n_neighbors=self.n_neighbors, metric=self.metric,
            p=self.p, metric_params=self.metric_params, mode=self.mode,
            include_self=False
            )
        Xt = Parallel(n_jobs=self.n_jobs)(delayed(_adjacency_matrix_func)(x)
                                          for x in Xt)

        return Xt
