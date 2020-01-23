"""Rescaling method for persistent homology."""
# License: GNU AGPLv3

import itertools

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class ConsistentRescaling(BaseEstimator, TransformerMixin):
    """Rescaling of distances between pairs of points by the geometric mean
    of the distances to the respective :math:`k`-th nearest neighbours.

    Based on ideas in [1]_. The computation during :meth:`transform` depends on
    the nature of the array `X`. If each entry in `X` along axis 0 represents a
    distance matrix :math:`D`, then the corresponding entry in the transformed
    array is the distance matrix
    :math:`D'_{ij} = D_{ij}/\\sqrt{D_{ik_i}D_{jk_j}}`, where :math:`k_i` is the
    index of the :math:`k`-th largest value in row :math:`i` (and similarly
    for :math:`j`). If the entries in `X` represent point clouds, their
    distance matrices are first computed, and then rescaled according to the
    same formula.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, each entry in `X` along axis 0 is
        interpreted to be a distance matrix. Otherwise, entries are
        interpreted as feature arrays, and `metric` determines a rule with
        which to calculate distances between pairs of instances (i.e. rows)
        in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan" or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    metric_params : dict, optional, default: ``{}``
        Additional keyword arguments for the metric function.

    neighbor_rank : int, optional, default: ``1``
        Rank of the neighbors used to modify the metric structure according
        to the "consistent rescaling" procedure.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.homology import ConsistentRescaling
    >>> X = np.array([[[0, 0], [1, 2], [5, 6]]])
    >>> cr = ConsistentRescaling()
    >>> X_rescaled = cr.fit_transform(X)
    >>> print(X_rescaled.shape)
    (1, 3, 3)

    See also
    --------
    VietorisRipsPersistence

    References
    ----------
    .. [1] T. Berry and T. Sauer, "Consistent manifold representation for
           topological data analysis"; *Foundations of data analysis* **1**,
           pp. 1--38, 2019; doi: `10.3934/fods.2019001
           <http://dx.doi.org/10.3934/fods.2019001>`_.

    """

    _hyperparameters = {'neighbor_rank': [int, (1, np.inf)]}

    # TODO: Consider using an immutable default value for metric_params.
    def __init__(self, metric='euclidean', metric_params={}, neighbor_rank=1,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.neighbor_rank = neighbor_rank
        self.n_jobs = n_jobs

    def _consistent_homology_distance(self, X):
        Xm = pairwise_distances(X, metric=self.metric, n_jobs=1,
                                **self.metric_params)

        indices_k_neighbor = np.argsort(Xm)[:, self.neighbor_rank]
        distance_k_neighbor = Xm[np.arange(X.shape[0]),
                                 indices_k_neighbor]

        # Only calculate metric for upper triangle
        Xc = np.zeros(Xm.shape)
        iterator = itertools.combinations(range(Xm.shape[0]), 2)
        for i, j in iterator:
            Xc[i, j] = Xm[i, j] / (np.sqrt(distance_k_neighbor[i] *
                                           distance_k_neighbor[j]))
        return Xc + Xc.T

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_points) or (n_samples, \
            n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            ``(n_points, n_points)``. Otherwise, each such entry will be
            interpreted as an array of ``n_points`` row vectors in
            ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        check_array(X, allow_nd=True)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each entry in the input data array X, find the metric structure
        after consistent rescaling and encodes it as a distance matrix. Then,
        arrange all results in a single ndarray of appropriate shape.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_points) or (n_samples, \
            n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            ``(n_points, n_points)``. Otherwise, each such entry will be
            interpreted as an array of ``n_points`` row vectors in
            ``n_dimensions``-dimensional space.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_points, n_points)
            Array containing (as entries along axis 0) the distance matrices
            after consistent rescaling.

        """
        # Check if fit had been called
        check_is_fitted(self, '_is_fitted')
        X = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._consistent_homology_distance)(X[i])
            for i in range(X.shape[0]))
        Xt = np.array(Xt)
        return Xt
