# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors.base import VALID_METRICS

from sklearn.utils._joblib import Parallel, delayed

import numpy as np
import gudhi as gd
from ripser import ripser


class VietorisRipsPersistence(BaseEstimator, TransformerMixin):
    """Transformer for the calculation of persistence diagrams resulting
    from Vietoris-Rips filtrations. Given a point cloud in Euclidean space
    or an abstract metric space encoded by a distance matrix, information
    about the appearance and disappearance of "topological holes"
    (technically, homology classes) of various dimensions and at different
    distance scales is summarised in the corresponding persistence diagram.

    Parameters
    ----------
    metric : string or callable, optional, default: 'euclidean'
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and ``metric`` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If ``metric`` is a string, it must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter, or a metric
        listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS, including "euclidean",
        "manhattan", or "cosine".
        If ``metric`` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in X as input, and return a value indicating
        the distance between them.

    max_edge_length : float, optional, default: np.inf
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter.
        Points whose distance is greater than this value will never be
        connected by an edge, and topological features at scales larger than
        this value will not be detected.

    homology_dimensions : list, optional, default: [0, 1]
        List of dimensions (non-negative integers). Topological holes of each
        of these dimensions will be detected.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=[0, 1], n_jobs=None):
        self.metric = metric
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
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
        return {'metric': self.metric, 'max_edge_length': self.max_edge_length,
                'homology_dimensions': self.homology_dimensions,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(metric):
        """A class method that checks whether the hyperparameters and the input
        parameters of the :meth:`fit` are valid.
        """
        implemented_metric_types = set(['precomputed'] + [met for i in VALID_METRICS.values() for met in i])

        if metric not in implemented_metric_types:
            raise ValueError('The metric %s is not supported' % metric)

    def _ripser_diagram(self, X, is_distance_matrix, metric):
        X_diagram = ripser(X[X[:, 0] != np.inf], distance_matrix=is_distance_matrix,
                           metric=metric, maxdim=max(self.homology_dimensions),
                           thresh=self.max_edge_length)['dgms']

        if 0 in self.homology_dimensions:
            X_diagram[0] = X_diagram[0][:-1, :]

        return {dimension: X_diagram[dimension]
                for dimension in self.homology_dimensions}

    def _pad_diagram(self, diagram, max_length_list):
        padList = [((0, max(0, max_length_list[i] - diagram[dimension].shape[0])),
                    (0, 0)) for i, dimension in enumerate(self.homology_dimensions)]
        return {dimension: np.pad(diagram[dimension], padList[i], 'constant')
                for i, dimension in enumerate(self.homology_dimensions)}

    def _stack_padded_diagrams(self, diagrams):
        stacked_diagrams = {dimension: np.stack([diagrams[i][dimension] for i in range(len(diagrams))], axis=0) for dimension in self.homology_dimensions}

        # for dimension in self.homology_dimensions:
        #     if stackedDiagrams[dimension].size == 0:
        #         del stackedDiagrams[dimension]
        return stacked_diagrams

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, n_points) or (n_samples, n_points, n_features)
            Input data. If ``metric == 'precomputed'``, the input should be an ndarray
            whose each entry along axis 0 is a distance matrix of shape
            (n_points, n_points). Otherwise, each such entry will be interpreted as
            an ndarray of n_points in Euclidean space of dimension n_features.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """

        self._validate_params(self.metric)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Computes, for each dimension in ``homology_dimensions`` and for each
        point cloud or distance matrix in X, the relevant persistence diagram as
        an array of pairs (b, d) -- one per persistent topological hole -- where
        b is the scale at which the topological hole first appears, and d the scale
        at which the same hole disappears.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, n_points) or (n_samples, n_points, n_features)
            Input data. If ``metric == 'precomputed'``, the input should be
            an ndarray whose each entry along axis 0 is a distance matrix of shape
            (n_points, n_points). Otherwise, each such entry will be interpreted as
            an ndarray of n_points in Euclidean space of dimension n_features.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : dict of int: ndarray or dict of int: list
            Dictionary whose keys are the integers in ``self.homology_dimensions``,
            and whose values are ndarrays of shape (n_samples, M_d, 2) where,
            if m_{d,i} is the number of persistent topological features in the relevant
            dimension d found in sample i, then M_d = max {m_{d,i}: i = 1, ..., n_samples}.
            If ``pad == False``, then each list has length n_samples and its i-th entry
            is an ndarrays of shape (m_{d,i}, 2).

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        is_distance_matrix = (self.metric == 'precomputed')

        X_transformed = Parallel(n_jobs=self.n_jobs)(delayed(self._ripser_diagram)(X[i, :, :], is_distance_matrix, self.metric)
                                                     for i in range(X.shape[0]))

        max_length_list = [max(1, np.max([X_transformed[i][dimension].shape[0] for i in range(len(X_transformed))]))
                           for dimension in self.homology_dimensions]
        X_transformed = Parallel(n_jobs=self.n_jobs)(delayed(self._pad_diagram)(X_transformed[i], max_length_list)
                                                     for i in range(len(X_transformed)))
        X_transformed = self._stack_padded_diagrams(X_transformed)

        return X_transformed


class CubicalPersistence(BaseEstimator, TransformerMixin):
    """Transformer for the calculation of persistence diagrams resulting from images.
    Given an image, information about the appearance and disappearance
    of "topological holes" (technically, homology classes) of various dimensions and
    at different distance scales is summarised in the corresponding persistence diagram.

    Parameters
    ----------
    max_edge_length : float, optional, default: np.inf
        Upper bound on the maximum value of the Cubical filtration parameter.
        Points whose manhattan distance is greater than this value will never be connected
        by an edge, and topological features at scales larger than this value will not be
        detected.

    homology_dimensions : list, optional, default: [0, 1]
        List of dimensions (non-negative integers). Topological holes of each of
        these dimensions will be detected.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless in
        a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    """

    def __init__(self, max_edge_length=np.inf, homology_dimensions=[0, 1],
                 n_jobs=None):
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
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
        return {'max_edge_length': self.max_edge_length, 'homology_dimensions': self.homology_dimensions,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(homology_dimensions, n_dimensions):
        """A class method that checks whether the hyperparameters and the input parameters
        of the :meth:`fit` are valid.
        """
        try:
            assert set(homology_dimensions).issubset(set(range(n_dimensions)))
        except AssertionError:
            raise ValueError('The homology_dimensions specified contains '
                             'element(s) that are not within the range 0 to '
                             'the dimension of the images.')

    def _gudhi_diagram(self, X):
        cubical_complex = gd.CubicalComplex(dimensions=X.shape, top_dimensional_cells=X.flatten(order="F"))
        diagram = cubical_complex.persistence(homology_coeff_field=2, min_persistence=0)

        return {dimension: np.array([diagram[i][1] for i in range(len(diagram))
                                     if diagram[i][0] == dimension and diagram[i][1][1] != np.inf]).reshape((-1, 2))
                for dimension in self.homology_dimensions}

    def _pad_diagram(self, diagram, max_length_list):
        pad_list = [((0, max(0, max_length_list[i] - diagram[dimension].shape[0])), (0, 0)) for i, dimension in enumerate(self.homology_dimensions)]

        return {dimension: np.pad(diagram[dimension], pad_list[i], 'constant')
                for i, dimension in enumerate(self.homology_dimensions)}

    def _stack_padded_diagrams(self, diagrams):
        stacked_diagrams = {dimension: np.stack([diagrams[i][dimension] for i in range(len(diagrams))], axis=0) for dimension in self.homology_dimensions}
        return stacked_diagrams

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y) or (n_samples, n_pixels_x, n_pixels_y, n_pixels_z)
            Input data. Array of 2D or 3D images.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._validate_params(self.homology_dimensions, len(X.shape) - 1)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Computes, for each dimension in ``homology_dimensions`` and for each
        image in X, the relevant persistence diagram as
        an array of pairs (b, d) -- one per persistent topological hole -- where
        b is the scale at which the topological hole first appears, and d the scale
        at which the same hole disappears.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y) or (n_samples, n_pixels_x, n_pixels_y, n_pixels_z)
            Input data. Array of 2D or 3D images.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : dict of int: ndarray or dict of int: list
            Dictionary whose keys are the integers in ``self.homology_dimensions``,
            and whose values are ndarrays if ``pad == True``, and lists otherwise.
            In the former case, the ndarrays have shape (n_samples, M_d, 2) where,
            if m_{d,i} is the number of persistent topological features in the relevant
            dimension d found in sample i, then M_d = max {m_{d,i}: i = 1, ..., n_samples}.
            If ``pad == False``, then each list has length n_samples and its i-th entry
            is an ndarrays of shape (m_{d,i}, 2).

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = Parallel(n_jobs=self.n_jobs)(delayed(self._gudhi_diagram)(X[i, :, :])
                                                     for i in range(X.shape[0]))
        max_length_list = [max(1, np.max([X_transformed[i][dimension].shape[0] for i in range(len(X_transformed))]))
                           for dimension in self.homology_dimensions]
        X_transformed = Parallel(n_jobs=self.n_jobs)(delayed(self._pad_diagram)(X_transformed[i], max_length_list)
                                                     for i in range(len(X_transformed)))
        X_transformed = self._stack_padded_diagrams(X_transformed)

        return X_transformed


class PersistentEntropy(BaseEstimator, TransformerMixin):
    """Transformer for the calculation of persistent entropy from collections of
    persistence diagrams. Given a generic persistence diagram consisting of
    birth-death pairs (b, d), its persistent entropy is simply the (base e) entropy
    of the collection of differences d - b, normalized by the sum of all such differences.

    Parameters
    ----------
    len_vector : int, optional, default: 8
        Used for performance optimization by exploiting numpy's vectorization capabilities.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless in
        a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    """
    def __init__(self, len_vector=8, n_jobs=None):
        self.len_vector = len_vector
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
        return {'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
        of the :meth:`fit` are valid.
        """
        pass

    def _persistent_entropy(self, X):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_normalized = X_lifespan / np.sum(X_lifespan, axis=1).reshape((-1, 1))
        return - np.sum(np.nan_to_num(X_normalized * np.log(X_normalized)), axis=1).reshape((-1, 1))

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers
            d representing homology dimensions, and whose values are ndarrays of
            shape (n_samples, M_d, 2) whose each entries along axis 0 are persistence
            diagrams with M_d persistent topological features. For example, X
            could be the result of applying the ``transform`` method of a
            ``VietorisRipsPersistence`` transformer to a collection of point
            clouds/distance matrices, but only if that transformer was instantiated
            with ``pad=True``.

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

    #@jit
    def transform(self, X, y=None):
        """For each key in the dictionary X and for each persistence diagram in the
        corresponding ndarray, computes that diagram's persistent entropy. All results
        are arranged into an ndarray of appropriate shape.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative integers
            d representing homology dimensions, and whose values are ndarrays of
            shape (n_samples, M_d, 2) whose each entries along axis 0 are persistence
            diagrams with M_d persistent topological features. For example, X
            could be the result of applying the ``transform`` method of a
            ``VietorisRipsPersistence`` transformer to a collection of point
            clouds/distance matrices.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_X_keys)
            Array of persistent entropies (one value per sample and per key in X).

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X[next(iter(X.keys()))].shape[0]
        n_dimensions = len(X.keys())

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs)(delayed(self._persistent_entropy)(X[dimension][slice_indices[i]: slice_indices[i + 1]])
                                                     for dimension in X.keys() for i in range(n_slices))

        X_transformed = np.hstack([np.concatenate([X_transformed[i * n_slices + j] for j in range(n_slices)], axis=0) for i in range(n_dimensions)])
        return X_transformed
