"""Persistent homology on point clouds or finite metric spaces."""
# License: GNU AGPLv3

from numbers import Real
from types import FunctionType

import numpy as np
from joblib import Parallel, delayed
from pyflagser import flagser_weighted
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_is_fitted

from ._utils import _postprocess_diagrams
from ..base import PlotterMixin
from ..externals.python import ripser, SparseRipsComplex, CechComplex
from ..plotting import plot_diagram
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params, check_point_clouds


@adapt_fit_transform_docs
class VietorisRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`Vietoris–Rips filtrations
    <vietoris-rips_complex_and_vietoris-rips_persistence>`.

    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Euclidean space, or an abstract :ref:`metric space
    <finite_metric_spaces_and_point_clouds>` encoded by a distance matrix,
    information about the appearance and disappearance of topological features
    (technically, :ref:`homology classes <homology_and_cohomology>`) of various
    dimensions and at different scales is summarised in the corresponding
    persistence diagram.

    **Important notes**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.
        - In homology dimension 0, :meth:`transform` automatically removes one
          birth-death pair whose death equals ``numpy.inf``.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices or of adjacency matrices of weighted
        undirected graphs. Otherwise, input data is to be interpreted as a
        collection of point clouds (i.e. feature arrays), and `metric`
        determines a rule with which to calculate distances between pairs of
        points (i.e. row vectors). If `metric` is a string, it must be one of
        the options allowed by :func:`scipy.spatial.distance.pdist` for its
        metric parameter, or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``'euclidean'``, ``'manhattan'`` or ``'cosine'``. If `metric` is a
        callable, it should take pairs of vectors (1D arrays) as input and, for
        each two vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Vietoris–Rips filtration parameter. Points whose
        distance is greater than this value will never be connected by an edge,
        and topological features at scales larger than this value will not be
        detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    FlagserPersistence, SparseRipsPersistence, WeakAlphaPersistence, \
    EuclideanCechPersistence, ConsistentRescaling, ConsecutiveRescaling

    Notes
    -----
    `Ripser <https://github.com/Ripser/ripser>`_ is used as a C++ backend for
    computing Vietoris–Rips persistent homology. Python bindings were modified
    for performance from the `ripser.py
    <https://github.com/scikit-tda/ripser.py>`_ package.

    References
    ----------
    [1] U. Bauer, "Ripser: efficient computation of Vietoris–Rips persistence \
        barcodes", 2019; `arXiv:1908.02518 \
        <https://arxiv.org/abs/1908.02518>`_.

    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'homology_dimensions': {
            'type': (list, tuple),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            },
        'coeff': {'type': int, 'in': Interval(2, np.inf, closed='left')},
        'max_edge_length': {'type': Real},
        'infinity_values': {'type': (Real, type(None))}
        }

    def __init__(self, metric='euclidean', homology_dimensions=(0, 1),
                 coeff=2, max_edge_length=np.inf, infinity_values=None,
                 n_jobs=None):
        self.metric = metric
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _ripser_diagram(self, X):
        Xdgms = ripser(X, maxdim=self._max_homology_dimension,
                       thresh=self.max_edge_length, coeff=self.coeff,
                       metric=self.metric)['dgms']

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][:-1, :]  # Remove one infinite bar in degree 0

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``'precomputed'``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays/sparse matrices.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``'precomputed'``, then:

                - if entries of `X` are dense, only their upper diagonal
                  portions (including the diagonal) are considered;
                - if entries of `X` are sparse, they do not need to be upper
                  diagonal or symmetric, but correct results can only be
                  guaranteed when only one between entry (i, j) and entry
                  (j, i) is stored, or both are stored but they are equal.
                - entries of `X` should be compatible with a filtration, i.e.
                  the value at index (i, j) should be no smaller than the
                  values at diagonal indices (i, i) and (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])
        self._is_precomputed = self.metric == 'precomputed'
        check_point_clouds(X, accept_sparse=True,
                           distance_matrices=self._is_precomputed)

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud or distance matrix in `X`, compute the
        relevant persistence diagram as an array of triples [b, d, q]. Each
        triple represents a persistent topological feature in dimension q
        (belonging to `homology_dimensions`) which is born at b and dies at d.
        Only triples in which b < d are meaningful. Triples in which b and d
        are equal ("diagonal elements") may be artificially introduced during
        the computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``'precomputed'``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays/sparse matrices.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``'precomputed'``, then:

                - if entries of `X` are dense, only their upper diagonal
                  portions (including the diagonal) are considered;
                - if entries of `X` are sparse, they do not need to be upper
                  diagonal or symmetric, but correct results can only be
                  guaranteed when only one between entry (i, j) and entry
                  (j, i) is stored, or both are stored but they are equal.
                - entries of `X` should be compatible with a filtration, i.e.
                  the value at index (i, j) should be no smaller than the
                  values at diagonal indices (i, i) and (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True,
                               distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._ripser_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class SparseRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`Sparse Vietoris–Rips filtrations
    <vietoris-rips_complex_and_vietoris-rips_persistence>`.

    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Euclidean space, or an abstract :ref:`metric space
    <finite_metric_spaces_and_point_clouds>` encoded by a distance matrix,
    information about the appearance and disappearance of topological features
    (technically, :ref:`homology classes <homology_and_cohomology>`) of various
    dimensions and at different scales is summarised in the corresponding
    persistence diagram.

    **Important notes**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.
        - In homology dimension 0, :meth:`transform` automatically removes one
          birth-death pair whose death equals ``numpy.inf``.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays), and
        `metric` determines a rule with which to calculate distances between
        pairs of instances (i.e. rows) in these arrays. If `metric` is a
        string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan", or "cosine". If `metric` is a
        callable, it is called on each pair of instances and the resulting
        value recorded. The callable should take two arrays from the entry in
        `X` as input, and return a value indicating the distance between them.

    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    epsilon : float between 0. and 1., optional, default: ``0.1``
        Parameter controlling the approximation to the exact Vietoris–Rips
        filtration. If set to `0.`, :class:`SparseRipsPersistence` leads to the
        same results as :class:`VietorisRipsPersistence` but is slower.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Sparse Rips filtration parameter. Points whose
        distance is greater than this value will never be connected by an edge,
        and topological features at scales larger than this value will not be
        detected.

    infinity_values : float or None, default : ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`. Set in :meth:`fit`.

    See also
    --------
    VietorisRipsPersistence, FlagserPersistence, WeakAlphaPersistence, \
    EuclideanCechPersistence, ConsistentRescaling, ConsecutiveRescaling

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing sparse Vietoris–Rips persistent homology. Python bindings
    were modified for performance.

    References
    ----------
    [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference \
        Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent_\
        cohomology.html>`_.

    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'homology_dimensions': {
            'type': (list, tuple),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            },
        'coeff': {'type': int, 'in': Interval(2, np.inf, closed='left')},
        'epsilon': {'type': Real, 'in': Interval(0, 1, closed='both')},
        'max_edge_length': {'type': Real},
        'infinity_values': {'type': (Real, type(None))}
        }

    def __init__(self, metric='euclidean', homology_dimensions=(0, 1),
                 coeff=2, epsilon=0.1, max_edge_length=np.inf,
                 infinity_values=None, n_jobs=None):
        self.metric = metric
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.epsilon = epsilon
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _gudhi_diagram(self, X):
        Xdgms = pairwise_distances(X, metric=self.metric)
        sparse_rips_complex = SparseRipsComplex(
            distance_matrix=Xdgms, max_edge_length=self.max_edge_length,
            sparse=self.epsilon)
        simplex_tree = sparse_rips_complex.create_simplex_tree(
            max_dimension=max(self._homology_dimensions) + 1)
        Xdgms = simplex_tree.persistence(
            homology_coeff_field=self.coeff, min_persistence=0)

        # Separate diagrams by homology dimensions
        Xdgms = {dim: np.array([Xdgms[i][1] for i in range(len(Xdgms))
                                if Xdgms[i][0] == dim]).reshape((-1, 2))
                 for dim in self.homology_dimensions}

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][1:, :]  # Remove one infinite bar in degree 0

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``'precomputed'``, and of distance matrices
            otherwise. Can be either a 3D ndarray whose zeroth dimension has
            size ``n_samples``, or a list containing ``n_samples`` 2D ndarrays.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``'precomputed'``, each entry of `X` should be
            compatible with a filtration, i.e. the value at index (i, j) should
            be no smaller than the values at diagonal indices (i, i) and
            (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])
        self._is_precomputed = self.metric == 'precomputed'
        check_point_clouds(X, accept_sparse=True,
                           distance_matrices=self._is_precomputed)

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud or distance matrix in `X`, compute the
        relevant persistence diagram as an array of triples [b, d, q]. Each
        triple represents a persistent topological feature in dimension q
        (belonging to `homology_dimensions`) which is born at b and dies at d.
        Only triples in which b < d are meaningful. Triples in which b and d
        are equal ("diagonal elements") may be artificially introduced during
        the computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``'precomputed'``, and of distance matrices
            otherwise. Can be either a 3D ndarray whose zeroth dimension has
            size ``n_samples``, or a list containing ``n_samples`` 2D ndarrays.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``'precomputed'``, each entry of `X` should be
            compatible with a filtration, i.e. the value at index (i, j) should
            be no smaller than the values at diagonal indices (i, i) and
            (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True,
                               distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._gudhi_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class WeakAlphaPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`weak alpha filtrations <TODO>`.

    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Euclidean space, information about the appearance and disappearance of
    topological features (technically, :ref:`homology classes
    <homology_and_cohomology>`) of various dimension and at different scales is
    summarised in the corresponding persistence diagram.

    The weak alpha filtration of a point cloud is defined to be the
    :ref:`Vietoris–Rips filtration
    <vietoris-rips_complex_and_vietoris-rips_persistence>` of the sparse matrix
    of Euclidean distances between neighbouring vertices in the Delaunay
    triangulation of the point cloud. In low dimensions, computing the
    persistent homology of this filtration can be much faster than computing
    Vietoris–Rips persistent homology via :class:`VietorisRipsPersistence`.


    **Important notes**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.
        - In homology dimension 0, :meth:`transform` automatically removes one
          birth-death pair whose death equals ``numpy.inf``.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Vietoris–Rips filtration parameter. Points whose
        distance is greater than this value will never be connected by an edge,
        and topological features at scales larger than this value will not be
        detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    VietorisRipsPersistence, FlagserPersistence, SparseRipsPersistence, \
    EuclideanCechPersistence

    Notes
    -----
    Delaunay triangulation are computed by :class:`scipy.spatial.Delaunay`.
    Ripser <https://github.com/Ripser/ripser>`_ is used as a C++ backend for
    computing Vietoris–Rips persistent homology. Python bindings were modified
    for performance from the `ripser.py
    <https://github.com/scikit-tda/ripser.py>`_ package.

    References
    ----------
    [1] U. Bauer, "Ripser: efficient computation of Vietoris–Rips persistence \
        barcodes", 2019; `arXiv:1908.02518 \
        <https://arxiv.org/abs/1908.02518>`_.

    """

    _hyperparameters = {
        'homology_dimensions': {
            'type': (list, tuple),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            },
        'coeff': {'type': int, 'in': Interval(2, np.inf, closed='left')},
        'max_edge_length': {'type': Real},
        'infinity_values': {'type': (Real, type(None))}
        }

    def __init__(self, homology_dimensions=(0, 1), coeff=2,
                 max_edge_length=np.inf, infinity_values=None, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _weak_alpha_diagram(self, X):
        # `indices` will serve as the array of column indices
        indptr, indices = Delaunay(X).vertex_neighbor_vertices

        # Compute the array of row indices
        row = np.zeros_like(indices)
        row[indptr[1:-1]] = 1
        np.cumsum(row, out=row)

        # We only need the upper diagonal
        mask = indices > row
        row, col = row[mask], indices[mask]
        dists = np.linalg.norm(X[row] - X[col], axis=1)
        # Note: passing the shape explicitly should not be needed in more
        # recent versions of C++ ripser
        n_points = len(X)
        dm = coo_matrix((dists, (row, col)), shape=(n_points, n_points))

        Xdgms = ripser(dm, maxdim=self._max_homology_dimension,
                       thresh=self.max_edge_length, coeff=self.coeff,
                       metric='precomputed')['dgms']

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][:-1, :]  # Remove one infinite bar

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])
        check_point_clouds(X)

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud in `X`, compute the relevant persistence
        diagram as an array of triples [b, d, q]. Each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Only triples
        in which b < d are meaningful. Triples in which b and d are equal
        ("diagonal elements") may be artificially introduced during the
        computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._weak_alpha_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class EuclideanCechPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    `Cech filtrations <TODO>`_.

    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Euclidean space, information about the appearance and disappearance of
    topological features (technically, :ref:`homology classes
    <homology_and_cohomology>`) of various dimensions and at different scales
    is summarised in the corresponding persistence diagram.

    **Important notes**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.
        - In homology dimension 0, :meth:`transform` automatically removes one
          birth-death pair whose death equals ``numpy.inf``.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Cech filtration parameter. Topological features at
        scales larger than this value will not be detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    VietorisRipsPersistence, FlagserPersistence, SparseRipsPersistence,
    WeakAlphaPersistence

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing Cech persistent homology. Python bindings were modified for
    performance.

    References
    ----------
    [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference \
        Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent_\
        cohomology.html>`_.

    """

    _hyperparameters = {
        'homology_dimensions': {
            'type': (list, tuple),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            },
        'coeff': {'type': int, 'in': Interval(2, np.inf, closed='left')},
        'max_edge_length': {'type': Real,
                            'in': Interval(0, np.inf, closed='right')},
        'infinity_values': {'type': (Real, type(None)),
                            'in': Interval(0, np.inf, closed='neither')},
        }

    def __init__(self, homology_dimensions=(0, 1), coeff=2,
                 max_edge_length=np.inf, infinity_values=None, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _gudhi_diagram(self, X):
        cech_complex = CechComplex(points=X, max_radius=self.max_edge_length)
        simplex_tree = cech_complex.create_simplex_tree(
            max_dimension=max(self._homology_dimensions) + 1)
        Xdgms = simplex_tree.persistence(
            homology_coeff_field=self.coeff, min_persistence=0)

        # Separate diagrams by homology dimensions
        Xdgms = {dim: np.array([Xdgms[i][1] for i in range(len(Xdgms))
                                if Xdgms[i][0] == dim]).reshape((-1, 2))
                 for dim in self.homology_dimensions}

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][1:, :]  # Remove one infinite bar in degree 0

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_point_clouds(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud in `X`, compute the relevant persistence
        diagram as an array of triples [b, d, q]. Each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Only triples
        in which b < d are meaningful. Triples in which b and d are equal
        ("diagonal elements") may be artificially introduced during the
        computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays in
            `X`. ``n_features`` equals :math:`\\sum_q n_q`, where :math:`n_q`
            is the maximum number of topological features in dimension
            :math:`q` across all samples in `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._gudhi_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class FlagserPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`filtrations <filtered_complex>` of :ref:`directed or undirected flag
    complexes <clique_and_flag_complexes>`.

    Given a weighted directed or undirected graph, information about the
    appearance and disappearance of topological features (technically,
    :ref:`homology classes <homology_and_cohomology>`) of various dimension and
    at different scales is summarised in the corresponding persistence diagram.

    **Important notes**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.
        - In homology dimension 0, :meth:`transform` automatically removes one
          birth-death pair whose death equals ``numpy.inf``.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    directed : bool, optional, default: ``True``
        If ``True``, :meth:`transform` computes the persistence diagrams of the
        filtered directed flag complexes arising from the input collection of
        weighted directed graphs. If ``False``, :meth:`transform` computes the
        persistence diagrams of the filtered undirected flag complexes obtained
        by regarding all input weighted graphs as undirected, and:

        - if `max_edge_weight` is ``numpy.inf``, it is sufficient to pass a
          collection of (dense or sparse) upper-triangular matrices;
        - if `max_edge_weight` is finite, it is recommended to pass either a
          collection of symmetric dense matrices, or a collection of sparse
          upper-triangular matrices.

    filtration : string, optional, default: ``'max'``
        Algorithm determining the filtration values of higher order simplices
        from the weights of the vertices and edges. Possible values are:
        ['dimension', 'zero', 'max', 'max3', 'max_plus_one', 'product', 'sum',
        'pmean', 'pmoment', 'remove_edges', 'vertex_degree']

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    max_edge_weight : float, optional, default: ``numpy.inf``
        Maximum edge weight to be considered in the filtration. All edge
        weights greater than this value will be considered as absent from the
        filtration and topological features at scales larger than this value
        will not be detected.

    infinity_values : float or None, default : ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_weight`. ``None`` means that this death
        value is declared to be equal to `max_edge_weight`.

    max_entries : int, optional, default: ``-1``
        Number controlling the degree of precision in the matrix reductions
        performed by the the backend. Corresponds to the parameter
        ``approximation`` in :func:`pyflagser.flagser_weighted` and
        :func:`pyflagser.flagser_unweighted`. Increase for higher precision,
        decrease for faster computation. A good value is often ``100000`` in
        hard problems. A negative value computes highest possible precision.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_weight`.

    See also
    --------
    VietorisRipsPersistence, SparseRipsPersistence, WeakAlphaPersistence,
    EuclideanCechPersistence, ConsistentRescaling, ConsecutiveRescaling

    Notes
    -----
    The `pyflagser <https://github.com/giotto-ai/pyflagser>`_ Python package
    is used for binding `Flagser <https://github.com/luetge/flagser>`_, a C++
    backend for computing the (persistent) homology of (filtered) directed
    flag complexes. For more details, please refer to the `flagser \
    documentation <https://github.com/luetge/flagser/blob/master/docs/\
    documentation_flagser.pdf>`_.

    References
    ----------
    [1] D. Luetgehetmann, D. Govc, J. P. Smith, and R. Levi, "Computing \
        persistent homology of directed flag complexes", Algorithms, 13(1), \
        2020.

    """

    _hyperparameters = {
        'homology_dimensions': {
            'type': (list, tuple),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            },
        'directed': {'type': bool},
        'coeff': {'type': int, 'in': Interval(2, np.inf, closed='left')},
        'max_edge_weight': {'type': Real},
        'infinity_values': {'type': (Real, type(None))},
        'max_entries': {'type': int}
        }

    def __init__(self, homology_dimensions=(0, 1), directed=True,
                 filtration='max', coeff=2, max_edge_weight=np.inf,
                 infinity_values=None, max_entries=-1, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.directed = directed
        self.filtration = filtration
        self.coeff = coeff
        self.max_edge_weight = max_edge_weight
        self.infinity_values = infinity_values
        self.max_entries = max_entries
        self.n_jobs = n_jobs

    def _flagser_diagram(self, X):
        Xdgms = [np.empty((0, 2), dtype=float)] * self._min_homology_dimension
        Xdgms += flagser_weighted(X, max_edge_weight=self.max_edge_weight,
                                  min_dimension=self._min_homology_dimension,
                                  max_dimension=self._max_homology_dimension,
                                  directed=self.directed,
                                  filtration=self.filtration, coeff=self.coeff,
                                  approximation=self.max_entries)['dgms']
        n_missing_dims = self._max_homology_dimension + 1 - len(Xdgms)
        if n_missing_dims:
            Xdgms += [np.empty((0, 2), dtype=float)] * n_missing_dims

        if 0 in self._homology_dimensions:
            Xdgms[0] = Xdgms[0][:-1, :]  # Remove one infinite bar in degree 0

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input collection of adjacency matrices of weighted directed or
            undirected graphs. Can be either a 3D ndarray whose zeroth
            dimension has size ``n_samples``, or a list containing
            ``n_samples`` 2D ndarrays/sparse matrices. In each adjacency
            matrix, diagonal elements are vertex weights and off-diagonal
            elements are edges weights. It is assumed that a vertex weight
            cannot be larger than the weight of the edges it
            forms. The way zero values are handled depends on the format of the
            matrix. If the matrix is a dense ``numpy.ndarray``, zero values
            denote zero-weighted edges. If the matrix is a sparse
            ``scipy.sparse`` matrix, explicitly stored off-diagonal zeros and
            all diagonal zeros denote zero-weighted edges. Off-diagonal values
            that have not been explicitly stored are treated by
            ``scipy.sparse`` as zeros but will be understood as
            infinitely-valued edges, i.e., edges absent from the filtration.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_point_clouds(X, accept_sparse=True, distance_matrices=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs',
                                                               'filtration'])

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_weight
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._min_homology_dimension = self._homology_dimensions[0]
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each adjacency matrix in `X`, compute the relevant persistence
        diagram as an array of triples [b, d, q]. Each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Only triples
        in which b < d are meaningful. Triples in which b and d are equal
        ("diagonal elements") may be artificially introduced during the
        computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input collection of adjacency matrices of weighted directed or
            undirected graphs. Can be either a 3D ndarray whose zeroth
            dimension has size ``n_samples``, or a list containing
            ``n_samples`` 2D ndarrays/sparse matrices. In each adjacency
            matrix, diagonal elements are vertex weights and off-diagonal
            elements are edges weights. It is assumed that a vertex weight
            cannot be larger than the weight of the edges it
            forms. The way zero values are handled depends on the format of the
            matrix. If the matrix is a dense ``numpy.ndarray``, zero values
            denote zero-weighted edges. If the matrix is a sparse
            ``scipy.sparse`` matrix, explicitly stored off-diagonal zeros and
            all diagonal zeros denote zero-weighted edges. Off-diagonal values
            that have not been explicitly stored are treated by
            ``scipy.sparse`` as zeros but will be understood as
            infinitely-valued edges, i.e., edges absent from the filtration.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True, distance_matrices=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._flagser_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )
