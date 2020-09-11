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
    """Adjacency matrices of k-nearest neighbor graphs.

    Given a two-dimensional array of row vectors seen as points in
    high-dimensional space, the corresponding kNN graph is a simple,
    undirected and unweighted graph with a vertex for every vector in the
    array, and an edge between two vertices whenever either the first
    corresponding vector is among the k nearest neighbors of the
    second, or vice-versa.

    :func:`sklearn.neighbors.kneighbors_graph` is used to compute the
    adjacency matrices of kNN graphs.

    Parameters
    ----------
    n_neighbors : int, optional, default: ``4``
        Number of neighbors to use.

    mode : ``'connectivity'`` | ``'distance'``, optional, \
        default: ``'connectivity'``
        Type of returned matrices: ``'connectivity'`` will return the 0-1
        connectivity matrices, and ``'distance'`` will return the distances
        between neighbors according to the given metric.

    metric : string or callable, optional, default: ``'euclidean'``
        Metric to use for distance computation. Any metric from scikit-learn
        or :mod:`scipy.spatial.distance` can be used.
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Distance matrices are not supported.
        Valid values for `metric` are:

        - from scikit-learn: [``'cityblock'``, ``'cosine'``, ``'euclidean'``,
          ``'l1'``, ``'l2'``, ``'manhattan'``]
        - from :mod:`scipy.spatial.distance`: [``'braycurtis'``,
          ``'canberra'``, ``'chebyshev'``, ``'correlation'``, ``'dice'``,
          ``'hamming'``, ``'jaccard'``, ``'kulsinski'``, ``'mahalanobis'``,
          ``'minkowski'``, ``'rogerstanimoto'``, ``'russellrao'``,
          ``'seuclidean'``, ``'sokalmichener'``, ``'sokalsneath'``,
          ``'sqeuclidean'``, ``'yule'``]

        See the documentation for :mod:`scipy.spatial.distance` for details on
        these metrics.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function.

    p : int, optional, default: ``2``
        Parameter for the Minkowski (i.e. :math:`\\ell^p`) metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. Only relevant
        when `metric` is ``'minkowski'``. `p` = 1 is the Manhattan distance,
        and `p` = 2 reduces to the Euclidean distance.

    metric_params : dict, optional, default: ``{}``
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

    """

    def __init__(self, mode='connectivity', n_neighbors=4, metric='euclidean',
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
            n_points, n_dimensions)
            Input data representing a collection of point clouds. Each entry
            in `X` is a 2D array of shape ``(n_points, n_dimensions)``.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_point_clouds(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute kNN graphs and return their adjacency matrices in sparse
        format.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_points, n_dimensions)
            Input data representing a collection of point clouds. Each entry
            in `X` is a 2D array of shape ``(n_points, n_dimensions)``.

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
        Xt = check_point_clouds(X)

        _adjacency_matrix_func = partial(
            kneighbors_graph, n_neighbors=self.n_neighbors, metric=self.metric,
            p=self.p, metric_params=self.metric_params, mode=self.mode,
            include_self=False
            )
        Xt = Parallel(n_jobs=self.n_jobs)(delayed(_adjacency_matrix_func)(x)
                                          for x in Xt)
        return Xt
