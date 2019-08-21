# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.neighbors import NearestNeighbors

import numpy as np
import scipy.sparse as sp


class TransitionGraph(BaseEstimator, TransformerMixin):
    """Given a collection of two-dimensional arrays, with row :math:`i` in
    array :math:`A` encoding the "state" of a system at "time" :math:`i`,
    this transformer returns a corresponding collection of so-called
    *transition graphs*. The vertex set of graph :math:`G` corresponding to
    :math:`A` is the set of all unique rows (states) in :math:`A`, and there
    is an edge between two vertices if and only if one of the two rows
    immediately follows the other anywhere in :math:`A`.

    Parameters
    ----------
    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Examples
    --------
    >>> from giotto.graphs import TransitionGraph
    >>> dyn = np.array([[['a'], ['b'], ['c']],
    ...                 [['c'], ['a'], ['b']])
    >>> tg = TransitionGraph()
    >>> tg.fit_transform(dyn)

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

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
        return {'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.
        """
        pass

    def _make_adjacency_matrix(self, X):
        indices = np.unique(X, axis=0, return_inverse=True)[1]
        n_indices = 2 * (len(indices) - 1)
        first = indices[:-1]
        second = indices[1:]
        A = sp.csr_matrix((np.full(n_indices, 1),
                           (np.concatenate([first, second]),
                            np.concatenate([second, first]))))
        sp.csr_matrix.setdiag(A, 0)
        return A

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_time_steps, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params()

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Create transition graphs from the input data and return their
        adjacency matrices. The graphs are simple, undirected and
        unweighted, and the adjacency matrices are sparse matrices of type
        bool.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_time_steps, n_features)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : array of sparse boolean matrices, shape (n_samples, )
            The collection of ``n_samples`` transition graphs. Each transition
            graph is encoded by a sparse matrix of boolean type.
        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs)(
            delayed(self._make_adjacency_matrix)(X[i]) for i in
            range(n_samples))
        X_transformed = np.array(X_transformed)
        return X_transformed


class KNeighborsGraph(BaseEstimator, TransformerMixin):
    r"""Calculates adjacency matrices of :math:`k`-nearest neighbor graphs.

    Let :math:`k` be a positive integer, and :math:`X` be a collection of point
    clouds in Euclidean space, each encoded as a two-dimensional array. For
    each point cloud :math:`\mathcal{P}` in :math:`X`, the corresponding kNN
    graph is an undirected and unweighted graph with an edge between any two
    points :math:`p_i, p_j` in :math:`\mathcal{P}` whenever either :math:`p_i`
    is among the :math:`k`-th nearest neighbors of :math:`p_j`, or :math:`p_j`
    is among the :math:`k`-th nearest neighbors of resp. :math:`p_i`. A point
    is not regarded as a neighbor of itself, i.e. the resulting graph is
    simple.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

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

    """

    def __init__(self, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='euclidean',
                 p=2, n_jobs=None, metric_params={}):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
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
        return {'n_neighbors': self.n_neighbors, 'radius': self.radius,
                'algorithm': self.algorithm,
                'leaf_size': self.leaf_size, 'metric': self.metric,
                'p': self.p, 'n_jobs': self.n_jobs,
                'metric_params': self.metric_params}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.
        """
        pass

    def _make_adjacency_matrix(self, X):
        self._nearest_neighbors.fit(X)
        A = self._nearest_neighbors.kneighbors_graph(
            X,
            n_neighbors=self.n_neighbors+1,
            mode='connectivity',
            include_self=False)
        rows, cols = A.nonzero()
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
        self._nearest_neighbors = NearestNeighbors(**nearest_neighbors_params)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute the adjacency matrix of the kNN graph of each entry in the
        input array along axis 0. Note: the method
        :meth:`sklearn.neighbors.kneighbors_graph` is used, but the set of
        :math:`k`-nearest neighbors of a point here corresponds to the set of
        (:math:`k+1`)-nearest neighbors according to the convention used there.

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


class GraphGeodesicDistance(BaseEstimator, TransformerMixin):
    """Given a collection of graphs presented as adjacency matrices,
    this transformer calculates for each graph the length of the shortest
    path between any of its two vertices. The result is a collection of
    dense distance matrices of variable size.

    Parameters
    ----------
    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

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
        return {'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:'fit' are valid.
        """
        pass

    def _geodesic_distance(self, X):
        X_distance = graph_shortest_path(X)
        X_distance[X_distance == 0] = np.inf  # graph_shortest_path returns a
        # float64 array, so inserting np.inf does not change the type.
        # Ideally however, graph_shortest_path would return an int array!
        np.fill_diagonal(X_distance, 0)
        return X_distance

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of sparse or dense arrays, shape (n_samples, )
            Input data, i.e. a collection of adjacency matrices of graphs.

        y : None
            There is no need of a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._validate_params()

        self._is_fitted = True
        return self

    # @jit
    def transform(self, X, y=None):
        """For each adjancency matrix in X, compute the lenghts of the graph
        shortest path between any two vertices, and arrange them in a
        distance matrix. The method
        :meth:`sklearn.utils.graph_shortest_path.graph_shortest_path` is used.

        Parameters
        ----------
        X : ndarray of sparse or dense arrays, shape (n_samples, )
            Input data, i.e. a collection of adjacency matrices of graphs.

        y : None
            Ignored.

        Returns
        -------
        X_transformed : ndarray of float, shape (n_samples, ) or
        (n_samples, n_vertices, n_vertices)
            Resulting array of distance matrices. If the distance matrices
            have variable size across samples, X is one-dimensional.

        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs)(
            delayed(self._geodesic_distance)(X[i]) for i in range(n_samples))
        X_transformed = np.array(X_transformed)
        return X_transformed
