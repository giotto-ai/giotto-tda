# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.utils.validation import check_is_fitted
from ..utils.validation import check_graph


class GraphGeodesicDistance(BaseEstimator, TransformerMixin):
    """Given a collection of graphs presented as sparse adjacency matrices,
    this transformer calculates for each graph the length of the shortest
    path between any of its two vertices. The result is a collection of
    dense distance matrices of variable size.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Examples
    --------
    >>> import numpy as np
    >>> from giotto.graphs import GraphGeodesicDistance
    >>> X = np.array([
    ...         np.array([
    ...             [0, 1, 3, 0, 0],
    ...             [1, 0, 5, 0, 0],
    ...             [3, 5, 0, 4, 0],
    ...             [0, 0, 4, 0, 0],
    ...             [0, 0, 0, 0, 0]])])
    >>> ggd = GraphGeodesicDistance()
    >>> ggd = ggd.fit(X)
    >>> print(ggd.transform(X)[0])
    [[ 0.  1.  3.  7. inf]
     [ 1.  0.  4.  8. inf]
     [ 3.  4.  0.  4. inf]
     [ 7.  8.  4.  0. inf]
     [inf inf inf inf  0.]]

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _geodesic_distance(self, X):
        X_distance = graph_shortest_path(X)
        X_distance[X_distance == 0] = np.inf  # graph_shortest_path returns a
        # float64 array, so inserting np.inf does not change the type.
        # Ideally however, graph_shortest_path would return an int array!
        np.fill_diagonal(X_distance, 0)
        return X_distance

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
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

        """
        X = check_graph(X)

        self._is_fitted = True
        return self

    # @jit
    def transform(self, X, y=None):
        """For each adjancency matrix in `X`, compute the lenghts of the graph
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
        Xt : ndarray of float, shape (n_samples, ) or
        (n_samples, n_vertices, n_vertices)
            Resulting array of distance matrices. If the distance matrices
            have variable size across samples, X is one-dimensional.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        X = check_graph(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._geodesic_distance)(X[i]) for i in range(X.shape[0]))
        Xt = np.array(Xt)
        return Xt
