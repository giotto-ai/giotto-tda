# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD


import numpy as np
from sklearn.utils._joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.utils.validation import check_is_fitted


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
