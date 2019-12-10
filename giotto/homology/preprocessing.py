"""Local point cloud extractor"""
# License: Apache 2.0


import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform, pdist
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from ..utils import validate_params


def _check_k_min_max(k_min, k_max):
    if k_min > k_max:
        raise ValueError("'k_min' should be less than 'k_max', but got {} and "
                         "{}".format(k_min, k_max))


class LocalNeighborhood(BaseEstimator, TransformerMixin):
    """For each entry of ``X``, compute the (local) point cloud associated to
    it by considering the neighborhood points according to the chosen
    ``metric``.
    The number of points per (local) point cloud is affected by three
    parameters: ``k_min`` and ``k_max`` represent respectively the minimum and
    maximum number of points, while the ``dist_percentage`` is the percentage
    of the diameter of the total point cloud that is equivalent to the radius
    in which the (local) point cloud is contained.
    Since this class is thought as a pre-process step before the persistence
    diagrams creation, in order to have (local) point clouds of the same
    length, all of them are padded with the last value in order to have length
    equal to the point cloud with the highest number of points, since this does
    not affect the persistence diagrams in any way.

    Parameters
    ----------
    dist_percentage : float, optional, default: ``0.05``
        The percentage of the radius of the total point cloud to consider when
        building the (local) point cloud for each entry of X.

    k_min : int, optional, default ``10``
        The minimum number of points to take when building the (local) point
        cloud.

    k_max : int, optional, default ``10``
        The maximum number of points to take when building the (local) point
        cloud.

    metric : str, optional, default: ``euclidean``
        The metric to use in order to compute the distance between points.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 7)
    >>> space_extraction = LocalNeighborhood(dist_percentage=0.3, k_min=2,
    ...                                      k_max=5)
    >>> Xt = space_extraction.fit_transform(X)
    >>> print(Xt.shape)
    (10, 2, 7)

    """
    _hyperparameters = {'dist_percentage': [float, (0, 1)],
                        'k_min': [int, (1, np.inf)],
                        'k_max': [int, (1, np.inf)]
                        }

    def __init__(self, dist_percentage=0.05, k_min=10, k_max=100,
                 metric='euclidean', n_jobs=None):
        self.dist_percentage = dist_percentage
        self.k_min = k_min
        self.k_max = k_max
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        check_array(X)
        _check_k_min_max(self.k_min, self.k_max)

        self._is_fitted = True

        return self

    def transform(self, X, y=None):
        """For each entry in the input data array ``X``, returns the point
        cloud surrounding it.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_points, n_features)
            Array containing the (local) point clouds associated with each
            entry of ``X``.

        """
        check_is_fitted(self, attributes=["_is_fitted"])
        check_array(X)

        distance_matrix = squareform(pdist(X, self.metric))

        local_point_clouds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_point_clouds)(X, distance_matrix, i)
            for i in range(len(X))
        )
        max_n_points = np.max([x.shape[0] for x in local_point_clouds])

        Xt_dims = list(X.shape)
        Xt_dims.insert(1, max_n_points)
        Xt = np.empty(Xt_dims)

        for i, point_cloud in enumerate(local_point_clouds):
            Xt[i, : len(point_cloud[0])] = point_cloud[0]
            Xt[i, len(point_cloud[0]):] = point_cloud[0][-1]

        return Xt

    def _extract_point_clouds(self, X, matrix_distances, ind_x):
        target_vector_dist = matrix_distances[ind_x]
        max_dist = np.max(target_vector_dist) * self.dist_percentage
        indices = target_vector_dist <= max_dist

        if np.sum(indices) > self.k_max:
            indices = np.argsort(target_vector_dist)[: self.k_max]
        elif np.sum(indices) < self.k_min:
            indices = np.argsort(target_vector_dist)[: self.k_min]

        return X[indices]
