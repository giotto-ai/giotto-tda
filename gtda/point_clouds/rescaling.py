"""Rescaling methods for persistent homology."""
# License: GNU AGPLv3

import itertools
from numbers import Real
from types import FunctionType

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import PlotterMixin
from ..plotting import plot_heatmap
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class ConsistentRescaling(BaseEstimator, TransformerMixin, PlotterMixin):
    """Rescaling of distances between pairs of points by the geometric mean
    of the distances to the respective :math:`k`-th nearest neighbours.

    Based on ideas in [1]_. The computation during :meth:`transform` depends on
    the nature of the array `X`. If each entry in `X` along axis 0 represents a
    distance matrix :math:`D`, then the corresponding entry in the transformed
    array is the distance matrix
    :math:`D'_{i,j} = D_{i,j}/\\sqrt{D_{i,k_i}D_{j,k_j}}`, where :math:`k_i` is
    the index of the :math:`k`-th largest value in row :math:`i` (and similarly
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

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function.

    neighbor_rank : int, optional, default: ``1``
        Rank of the neighbors used to modify the metric structure according
        to the "consistent rescaling" procedure.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params`.
        If `metric_params` is ``None``, it is set to the empty dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.point_clouds import ConsistentRescaling
    >>> X = np.array([[[0, 0], [1, 2], [5, 6]]])
    >>> cr = ConsistentRescaling()
    >>> X_rescaled = cr.fit_transform(X)
    >>> print(X_rescaled.shape)
    (1, 3, 3)

    See also
    --------
    ConsecutiveRescaling

    References
    ----------
    .. [1] T. Berry and T. Sauer, "Consistent manifold representation for
           topological data analysis"; *Foundations of data analysis* **1**,
           pp. 1--38, 2019; doi: `10.3934/fods.2019001
           <http://dx.doi.org/10.3934/fods.2019001>`_.

    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'metric_params': {'type': (dict, type(None))},
        'neighbor_rank': {
            'type': int, 'in': Interval(1, np.inf, closed='left')}
    }

    def __init__(self, metric='euclidean', metric_params=None, neighbor_rank=1,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.neighbor_rank = neighbor_rank
        self.n_jobs = n_jobs

    def _consistent_rescaling(self, X):
        Xm = pairwise_distances(X, metric=self.metric, n_jobs=1,
                                **self.effective_metric_params_)

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
        """Calculate :attr:`effective_metric_params_`. Then, return the
        estimator.

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
        check_array(X, allow_nd=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        return self

    def transform(self, X, y=None):
        """For each entry in the input data array X, find the metric structure
        after consistent rescaling and encode it as a distance matrix.

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
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._consistent_rescaling)(x) for x in Xt)
        Xt = np.array(Xt)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='blues'):
        """Plot a sample from a collection of distance matrices.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, n_points)
            Collection of distance matrices, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        colorscale : str, optional, default: ``'blues'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        """
        return plot_heatmap(Xt[sample], colorscale=colorscale)


@adapt_fit_transform_docs
class ConsecutiveRescaling(BaseEstimator, TransformerMixin, PlotterMixin):
    """Rescaling of distances between consecutive pairs of points by a fixed
    factor.

    The computation during :meth:`transform` depends on the nature of the array
    `X`. If each entry in `X` along axis 0 represents a distance matrix
    :math:`D`, then the corresponding entry in the transformed array is the
    distance matrix :math:`D'_{i,i+1} = \\alpha D_{i,i+1}` where
    :math:`\\alpha` is a positive factor. If the entries in `X` represent point
    clouds, their distance matrices are first computed, and then rescaled
    according to the same formula.

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

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function.

    factor : float, optional, default: ``0.``
        Factor by which to multiply the distance between consecutive
        points.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params`.
        If `metric_params` is ``None``, it is set to the empty dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.point_clouds import ConsecutiveRescaling
    >>> X = np.array([[[0, 0], [1, 2], [5, 6]]])
    >>> cr = ConsecutiveRescaling()
    >>> X_rescaled = cr.fit_transform(X)
    >>> print(X_rescaled.shape)
    (1, 3, 3)

    See also
    --------
    ConsistentRescaling

    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'metric_params': {'type': (dict, type(None))},
        'factor': {
            'type': Real, 'in': Interval(0, np.inf, closed='both')}
    }

    def __init__(self, metric='euclidean', metric_params=None, factor=0.,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.factor = factor
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Calculate :attr:`effective_metric_params_`. Then, return the
        estimator.

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
        check_array(X, allow_nd=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        return self

    def transform(self, X, y=None):
        """For each entry in the input data array X, find the metric structure
        after consecutive rescaling and encode it as a distance matrix.

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
            after consecutive rescaling.

        """
        check_is_fitted(self)
        is_precomputed = self.metric == 'precomputed'
        X = check_array(X, allow_nd=True, copy=is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(pairwise_distances)(
                x, metric=self.metric, n_jobs=1,
                **self.effective_metric_params_)
            for x in X)

        if is_precomputed:
            # Parallel loop above serves only as additional input validation
            Xt = X
        else:
            Xt = np.array(Xt)
        Xt[:, range(Xt.shape[1] - 1), range(1, Xt.shape[1])] *= self.factor
        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='blues'):
        """Plot a sample from a collection of distance matrices.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, n_points)
            Collection of distance matrices, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        colorscale : str, optional, default: ``'blues'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        """
        return plot_heatmap(Xt[sample], colorscale=colorscale)
