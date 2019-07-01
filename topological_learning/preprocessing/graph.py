import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.neighbors import NearestNeighbors

import numpy as np
import scipy.sparse as sp


class UniqueGraphEmbedder(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    samplingType : str
        The type of sampling

        - data_type: string, must equal either 'points' or 'distance_matrix'.
        - data_iter: an iterator. If data_iter is 'points' then each object in the iterator
          should be a numpy array of dimension (number of points, number of coordinates),
          or equivalent nested list structure. If data_iter is 'distance_matrix' then each
          object in the iterator should be a full (symmetric) square matrix (numpy array)
          of shape (number of points, number of points), __or a sparse distance matrix

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def _embed(self, X):
        indices = np.unique(X, axis=0, return_inverse=True)[1]
        n_indices = 2 * (len(indices) - 1)
        X_embedded = sp.csr_matrix( (np.full(n_indices, True), (np.concatenate([indices[:-1], indices[1:]]),
                                                                np.concatenate([indices[1:], indices[:-1]]))),
                                    dtype=bool)
        return X_embedded

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params()

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        """ Implementation of the sk-learn transform function that samples the input.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted'])


        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._embed) (X[i]) for i in range(n_samples) )
        X_transformed = np.array(X_transformed)
        return X_transformed

class NearestNeighborGraphEmbedder(BaseEstimator, TransformerMixin):
    """A graph embedder based on k-Nearest Neighbors algorithm.

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

    Examples
    --------
      >>> import numpy as np
      >>> from sklearn.neighbors import NearestNeighbors
      >>> samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
      >>> neigh = NearestNeighbors(2, 0.4)
      >>> neigh.fit(samples)  #doctest: +ELLIPSIS
      NearestNeighbors(...)
      >>> neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
      ... #doctest: +ELLIPSIS
      array([[2, 0]]...)
      >>> nbrs = neigh.radius_neighbors([[0, 0, 1.3]], 0.4, return_distance=False)
      >>> np.asarray(nbrs[0][0])
      array(2)

    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    KNeighborsRegressor
    RadiusNeighborsRegressor
    BallTree
    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.
    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """
    def __init__(self, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='euclidean',
                 p=2, mode='connectivity', n_jobs=None, metric_params={}):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.mode = mode
        self.n_jobs = n_jobs
        self.metric_params = metric_params

    def get_params(self, deep=True):
        return {'n_neighbors': self.n_neighbors, 'radius': self.radius, 'algorithm': self.algorithm,
                'leaf_size': self.leaf_size, 'metric': self.metric, 'p': self.p, 'mode': self.mode,
                'n_jobs': self.n_jobs, 'metric_params': self.metric_params}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        pass

    def _embed(self, X):
        self.nearest_neighbors.fit(X)
        X_embedded = self.nearest_neighbors.kneighbors_graph(X, n_neighbors=self.n_neighbors, mode=self.mode)
        sp.csr_matrix.setdiag(X_embedded, 0)
        rows, cols = X_embedded.nonzero()
        X_embedded[cols, rows] = X_embedded[rows, cols]
        return X_embedded


    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
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
        nearest_neighbors_params.pop('mode')
        self.nearest_neighbors = NearestNeighbors(**nearest_neighbors_params)

        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        """ Implementation of the sk-learn transform function that samples the input.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted'])


        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._embed) (X[i]) for i in range(n_samples) )
        X_transformed = np.array(X_transformed)
        return X_transformed


class GeodesicDistance(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
            of the :meth:'fit' are valid.
            """
        pass

    def _geodesic_distance(self, X):
        X_distance = graph_shortest_path(X)
        X_distance[X_distance == 0] = np.inf
        np.fill_diagonal(X_distance, 0)
        return X_distance

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

            Parameters
            ----------
            X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
            y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

            Returns
            -------
            self : object
            Returns self.
            """
        self._validate_params()

        self.is_fitted = True
        return self

    #@jit
    def transform(self, X, y=None):
        """ Implementation of the sk-learn transform function that samples the input.

            Parameters
            ----------
            X : array-like of shape = [n_samples, n_features]
            The input samples.

            Returns
            -------
            X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
            """
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted'])

        n_samples = X.shape[0]

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._geodesic_distance) (X[i]) for i in range(n_samples) )
        X_transformed = np.array(X_transformed)
        return X_transformed
