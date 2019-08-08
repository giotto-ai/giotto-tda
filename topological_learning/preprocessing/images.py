# authors: Adelie Garin <adelie.garin@epfl.ch>
#          Guillaume Tauzin <guillaume.tauzin@epfl.ch>

import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

import numpy as np
from scipy import ndimage as ndi


class HeightFiltration(BaseEstimator, TransformerMixin):
    """Transformer returning a representation of a collection (typically, a time series)
    of point clouds in :math:`\\mathbb{R}^d` -- where each point cloud is an array
    of size (n_points, d) -- as a collection of arrays of the same shape, whose
    each row is the result of applying ``np.argsort`` to the corresponding row
    in the point cloud array. Based on ideas in `arXiv:1904.07403 <https://arxiv.org/abs/1904.07403>`_.

    Parameters
    ----------
    direction : ndarray, required
        Direction of the height filtration.

    len_vector : int, optional, default: 8
        Used for performance optimization by exploiting numpy's vectorization capabilities.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless in
        a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    """

    def __init__(self, direction=None, len_vector=8, n_jobs=None):
        self.direction = direction
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
        return {'direction': self.direction, 'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(dimension_image, dimension_direction):
        """A class method that checks whether the hyperparameters and the input parameters
        of the :meth:'fit' are valid.
        """
        try:
            assert dimension_image == dimension_direction
        except:
            raise ValueError("The dimension of the direction vector does not correspond to the dimension of the image.")

    def _calculate_filtration(self, X):
        X_height = np.full(X.shape, np.inf, dtype=np.float)
        norm = np.linalg.norm(X.shape[1:])

        for i in range(X_height.shape[0]):
            X_height[i][X[i] == True] = np.dot(self._mesh[X[i] == True], self._direction).reshape((-1,))

        X_height = X_height / norm
        return X_height

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._dimension = len(X.shape) - 1
        self._validate_params(self._dimension, len(self.direction))

        self._direction = self.direction / np.linalg.norm(self.direction)

        mesh_range_list = [ np.arange(1, X.shape[i]+1) if self.direction[i-1] > 0
                            else -np.flip(np.arange(1, X.shape[i]+1)) for i in range(1, self._dimension+1) ]

        self._mesh = np.stack(np.meshgrid(*mesh_range_list, indexing='ij'), axis=self._dimension)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each point cloud in X, argsort each row, in ascending order.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray of int, shape (n_samples, n_points, d)
            The transformed array.

        """

        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._calculate_filtration) (X[slice_indices[i] : slice_indices[i+1]])
                                                       for i in range(n_slices) )
        X_transformed = np.concatenate(X_transformed)
        return X_transformed


class ThickeningFiltration(BaseEstimator, TransformerMixin):
    """Transformer returning a representation of a collection (typically, a time series)
    of point clouds in :math:`\\mathbb{R}^d` -- where each point cloud is an array
    of size (n_points, d) -- as a collection of arrays of the same shape, whose
    each row is the result of applying ``np.argsort`` to the corresponding row
    in the point cloud array. Based on ideas in `arXiv:1904.07403 <https://arxiv.org/abs/1904.07403>`_.

    Parameters
    ----------
    direction : ndarray, required
        Direction of the height filtration.

    len_vector : int, optional, default: 8
        Used for performance optimization by exploiting numpy's vectorization capabilities.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless in
        a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    """

    def __init__(self, n_iterations=None, len_vector=8, n_jobs=None):
        self.n_iterations = n_iterations
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
        return {'n_iterations': self.n_iterations, 'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
        of the :meth:'fit' are valid.
        """
        pass

    def _calculate_filtration(self, X):
        X_thick = X*1.
        for _ in range(self.n_iterations):
            X_thick += np.asarray([ndi.binary_dilation(X_thick[i]) for i in range(X_thick.shape[0])])

        mask_filtered = X_thick == 0
        X_thick = np.amax(X_thick, axis=tuple(range(1, self._dimension+1)), keepdims=True) - X_thick + 1
        X_thick[mask_filtered] = np.inf
        X_thick = (X_thick - 1) / self.n_iterations
        return X_thick

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._dimension = len(X.shape) - 1
        self._validate_params()

        if self.n_iterations is None:
            self.n_iterations = max(X.shape[1:])

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each point cloud in X, argsort each row, in ascending order.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray of int, shape (n_samples, n_points, d)
            The transformed array.

        """

        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._calculate_filtration) (X[slice_indices[i] : slice_indices[i+1]])
                                                       for i in range(n_slices) )
        X_transformed = np.concatenate(X_transformed)

        return X_transformed


class RadialFiltration(BaseEstimator, TransformerMixin):
    """Transformer returning a representation of a collection (typically, a time series)
    of point clouds in :math:`\\mathbb{R}^d` -- where each point cloud is an array
    of size (n_points, d) -- as a collection of arrays of the same shape, whose
    each row is the result of applying ``np.argsort`` to the corresponding row
    in the point cloud array. Based on ideas in `arXiv:1904.07403 <https://arxiv.org/abs/1904.07403>`_.

    Parameters
    ----------
    center : ndarray, required
        Center of the radial filtration.

    metric : string or callable, optional, default: 'euclidean'
        If set to ``'precomputed'``, each entry in X along axis 0 is interpreted to
        be a distance matrix. Otherwise, entries are interpreted as feature arrays,
        and ``metric`` determines a rule with which to calculate distances between
        pairs of instances (i.e. rows) in these arrays.
        If ``metric`` is a string, it must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter, or a metric listed
        in pairwise.PAIRWISE_DISTANCE_FUNCTIONS, including "euclidean", "manhattan",
        or "cosine"
        If ``metric`` is a callable function, it is called on each pair of instances
        and the resulting value recorded. The callable should take two arrays from
        the entry in X as input, and return a value indicating the distance between them.

    metric_params : dict, optional, default: {}
        Additional keyword arguments for the metric function.

    len_vector : int, optional, default: 8
        Used for performance optimization by exploiting numpy's vectorization capabilities.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless in
        a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    Attirbutes
    ----------
    _distance_center : ndarray
        Image of the distance to the center.

    """

    def __init__(self, center=None, max_radius=np.inf, metric='euclidean', metric_params={}, len_vector=8, n_jobs=None):
        self.center = center
        self.max_radius = max_radius
        self.metric = metric
        self.metric_params = metric_params
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
        return {'center': self.center,'max_radius': self.max_radius,
                'metric': self.metric, 'metric_params': self.metric_params,
                'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(center, dimension_image):
        """A class method that checks whether the hyperparameters and the input parameters
        of the :meth:'fit' are valid.
        """
        try:
            assert dimension_image == center.shape[0]
        except:
            raise ValueError("Please pass a center 1D array which length corresponds to the dimension of the image.")

    def _calculate_filtration(self, X):
        X_rad = np.nan_to_num(self._distance_center * X, nan=np.inf, posinf=np.inf)
        X_rad = np.nan_to_num(X_rad, posinf=-1)
        norm =  np.amax(X_rad, axis=tuple(range(1, self._dimension+1)), keepdims=True)
        X_rad = X_rad / norm
        X_rad[X_rad == -1] = np.inf
        X_rad[X == False] = np.inf
        return X_rad

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._dimension = len(X.shape) - 1
        self._validate_params(self.center, self._dimension)

        mesh_range_list = [ np.arange(0, X.shape[i]) for i in range(self._dimension, 0, -1) ]

        mesh = np.stack(np.meshgrid(*mesh_range_list), axis=self._dimension).reshape((-1, self._dimension))

        self._distance_center = pairwise_distances(self.center.reshape((1, -1)), mesh, metric=self.metric,
                                                   n_jobs=self.n_jobs, **self.metric_params).reshape(X.shape[1:])
        self._distance_center[self._distance_center > self.max_radius] = np.inf

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each point cloud in X, argsort each row, in ascending order.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, d)
            Input data. Each entry along axis 0 is interpreted as a point cloud
            in d-dimensional Euclidean space.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray of int, shape (n_samples, n_points, d)
            The transformed array.

        """

        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        n_samples = X.shape[0]

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._calculate_filtration) (X[slice_indices[i] : slice_indices[i+1]])
                                                       for i in range(n_slices) )
        X_transformed = np.concatenate(X_transformed)
        return X_transformed
