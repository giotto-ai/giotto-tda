# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
#          Philippe Nguyen <p.nguyen@l2f.ch>
# License: TBD

import numpy as np
from functools import partial
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._joblib import Parallel, delayed
from sklearn.neighbors import kneighbors_graph


class KNeighborsGraph(BaseEstimator, TransformerMixin):
    r"""Calculates adjacency matrices of :math:`k`-nearest neighbor graphs.

    Let :math:`k` be a positive integer, and :math:`X` be a collection of point
    clouds in Euclidean space, each encoded as a two-dimensional array. For
    each point cloud :math:`\mathcal{P}` in :math:`X`, the corresponding kNN
    graph is an undirected and unweighted graph with an edge between any two
    points :math:`p_i, p_j` in :math:`\mathcal{P}` whenever either :math:`p_i`
    is among the :math:`k` nearest neighbors of :math:`p_j`, or :math:`p_j`
    is among the :math:`k` nearest neighbors of resp. :math:`p_i`. A point
    is not regarded as a neighbor of itself, i.e. the resulting graph is
    simple.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Distance matrices are not supported.
        Valid values for metric are:
        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']
        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Examples
    --------
    >>> import numpy as np
    >>> from giotto.graphs import KNeighborsGraph
    >>> X = np.array([
    ...         np.array([
    ...             [0, 1, 3, 0, 0],
    ...             [1, 0, 5, 0, 0],
    ...             [3, 5, 0, 4, 0],
    ...             [0, 0, 4, 0, 0],
    ...             [0, 0, 0, 0, 0]])])
    >>> kng = KNeighborsGraph(n_neighbors=2)
    >>> kng = kng.fit(X)
    >>> print(kng.transform(X)[0].toarray())
    [[0. 1. 1. 1. 1.]
     [1. 0. 0. 1. 0.]
     [1. 0. 0. 0. 1.]
     [1. 1. 0. 0. 1.]
     [1. 0. 1. 1. 0.]]

    """

    def __init__(self, n_neighbors=5, metric='euclidean',
                 p=2, n_jobs=None, metric_params=None):
        if metric_params is None:
            metric_params = {}
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.metric_params = metric_params

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default: True
            Behaviour not yet implemented.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return {'n_neighbors': self.n_neighbors, 'metric': self.metric,
                'p': self.p, 'n_jobs': self.n_jobs,
                'metric_params': self.metric_params}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.
        """
        pass

    def _knn_graph(self, nearest_neighbors_params):
        return partial(kneighbors_graph, **nearest_neighbors_params,
                       mode='connectivity', include_self=False)

    def _make_adjacency_matrix(self, X):
        A = self._nearest_neighbors(X)
        rows, cols = A.nonzero()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SparseEfficiencyWarning)
            A[cols, rows] = 1
        return A

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, n_features)
            Input data. Each entry of X along axis 0 is interpreted as an
            ndarray of n_points in Euclidean space of dimension n_features.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._validate_params()

        nearest_neighbors_params = self.get_params()
        self._nearest_neighbors = self._knn_graph(nearest_neighbors_params)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute the adjacency matrix of the kNN graph of each entry in the
        input array along axis 0. The method
        :meth:`sklearn.neighbors.kneighbors_graph` is used.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, n_features)
            Input data. Each entry of X along axis 0 is interpreted as an
            ndarray of n_points in Euclidean space of dimension n_features.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray of sparse matrices in CSR format, shape
        (n_samples, )
            The transformed array.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs)(
            delayed(self._make_adjacency_matrix)(X[i]) for i in
            range(n_samples))
        X_transformed = np.array(X_transformed)
        return X_transformed
