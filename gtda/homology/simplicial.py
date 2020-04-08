"""Persistent homology on point clouds or finite metric spaces."""
# License: GNU AGPLv3

from numbers import Real
from types import FunctionType

import numpy as np
from joblib import Parallel, delayed
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
    Euclidean space, or an abstract
    :ref:`metric space <finite_metric_spaces_and_point_clouds>` encoded by a
    distance matrix, information about the appearance and disappearance of
    topological features (technically,
    :ref:`homology classes <homology_and_cohomology>`) of various dimension
    and at different scales is summarised in the corresponding persistence
    diagram.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices or of adjacency matrices of weighted
        undirected graphs. Otherwise, input data is to be interpreted as a
        collection of point clouds (i.e. feature arrays), and `metric`
        determines a rule with which to calculate distances between pairs of
        points (i.e. row vectors). If `metric` is a string, it must be one
        of the options allowed by :func:`scipy.spatial.distance.pdist` for
        its metric parameter, or a metric listed in
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
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris–Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this
        death value is declared to be equal to `max_edge_length`.

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
    SparseRipsPersistence, EuclideanCechPersistence, CubicalPersistence,
    ConsistentRescaling

    Notes
    -----
    `Ripser <https://github.com/Ripser/ripser>`_ is used as a C++ backend
    for computing Vietoris–Rips persistent homology. Python bindings were
    modified for performance from the `ripser.py
    <https://github.com/scikit-tda/ripser.py>`_ package.

    Persistence diagrams produced by this class must be interpreted with
    care due to the presence of padding triples which carry no information.
    See :meth:`transform` for additional information.

    References
    ----------
    [1] U. Bauer, "Ripser: efficient computation of Vietoris–Rips persistence \
        barcodes", 2019; `arXiv:1908.02518 \
        <https://arxiv.org/abs/1908.02518>`_.

    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'homology_dimensions': {
            'type': (list, tuple), 'of': {
                'type': int, 'in': Interval(0, np.inf, closed='left')}},
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
            Xdgms[0] = Xdgms[0][:-1, :]  # Remove one infinite bar

        # Add dimension as the third elements of each (b, d) tuple
        Xdgms = {dim: np.hstack([Xdgms[dim],
                                 dim * np.ones((Xdgms[dim].shape[0], 1),
                                               dtype=Xdgms[dim].dtype)])
                 for dim in self._homology_dimensions}
        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list
            Input data representing a collection of point clouds if `metric`
            was not set to ``'precomputed'``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays. If `metric` was
            set to ``'precomputed'``, each entry of `X` must be a square
            array and should be compatible with a filtration, i.e. the value
            at index (i, j) should be no smaller than the values at diagonal
            indices (i, i) and (j, j).

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
        check_point_clouds(X, distance_matrices=self._is_precomputed)

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
        X : ndarray or list
            Input data representing a collection of point clouds if `metric`
            was not set to ``'precomputed'``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays. If `metric` was
            set to ``'precomputed'``, each entry of `X` must be a square
            array and should be compatible with a filtration, i.e. the value
            at index (i, j) should be no smaller than the values at diagonal
            indices (i, i) and (j, j).

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
        X = check_point_clouds(X, distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._ripser_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions)


@adapt_fit_transform_docs
class SparseRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`Sparse Vietoris–Rips filtrations
    <vietoris-rips_complex_and_vietoris-rips_persistence>`.

    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Euclidean space, or an abstract
    :ref:`metric space <finite_metric_spaces_and_point_clouds>`
    encoded by a distance matrix, information about the appearance and
    disappearance of topological features (technically,
    :ref:`homology classes <homology_and_cohomology>`) of various dimensions
    and at different scales is summarised in the corresponding persistence
    diagram.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and `metric` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan", or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    epsilon : float between 0. and 1., optional, default: ``0.1``
        Parameter controlling the approximation to the exact Vietoris–Rips
        filtration. If set to `0.`, :class:`SparseRipsPersistence` leads to
        the same results as :class:`VietorisRipsPersistence` but is slower.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris–Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    infinity_values : float or None, default : ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this
        death value is declared to be equal to `max_edge_length`.

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
    VietorisRipsPersistence, EuclideanCechPersistence, CubicalPersistence,
    ConsistentRescaling

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing sparse Vietoris–Rips persistent homology. Python bindings
    were modified for performance.

    Persistence diagrams produced by this class must be interpreted with
    care due to the presence of padding triples which carry no information.
    See :meth:`transform` for additional information.

    References
    ----------
    [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference \
        Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent_\
        cohomology.html>`_.

    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'homology_dimensions': {
            'type': (list, tuple), 'of': {
                'type': int, 'in': Interval(0, np.inf, closed='left')}},
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
            Xdgms[0] = Xdgms[0][1:, :]  # Remove one infinite bar

        # Add dimension as the third elements of each (b, d) tuple
        Xdgms = {dim: np.hstack([Xdgms[dim],
                                 dim * np.ones((Xdgms[dim].shape[0], 1),
                                               dtype=Xdgms[dim].dtype)])
                 for dim in self._homology_dimensions}
        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list
            Input data representing a collection of point clouds or of distance
            matrices. Can be either a 3D ndarray whose zeroth dimension has
            size ``n_samples``, or a list containing ``n_samples`` 2D ndarrays.
            If ``metric == 'precomputed'``, elements of `X` must be square
            arrays representing distance matrices; otherwise, their rows are
            interpreted as vectors in Euclidean space and, when `X` is a list,
            warnings are issued when the number of columns (dimension of the
            Euclidean space) differs among samples.

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
        check_point_clouds(X, distance_matrices=self._is_precomputed)

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
        X : ndarray or list
            Input data representing a collection of point clouds or of distance
            matrices. Can be either a 3D ndarray whose zeroth dimension has
            size ``n_samples``, or a list containing ``n_samples`` 2D ndarrays.
            If ``metric == 'precomputed'``, elements of `X` must be square
            arrays representing distance matrices; otherwise, their rows are
            interpreted as vectors in Euclidean space and, when `X` is a list,
            warnings are issued when the number of columns (dimension of the
            Euclidean space) differs among samples.

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
        X = check_point_clouds(X, distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._gudhi_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(Xt, self._homology_dimensions,
                                   self.infinity_values_, self.n_jobs)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions)


@adapt_fit_transform_docs
class EuclideanCechPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    `Cech filtrations <TODO>`_.

    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Euclidean space, information about the appearance and disappearance of
    topological features (technically,
    :ref:`homology classes <homology_and_cohomology>`) of various dimensions
    and at different scales is summarised in the corresponding persistence
    diagram.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris–Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

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
    VietorisRipsPersistence, SparseRipsPersistence, CubicalPersistence,
    ConsistentRescaling

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing Cech persistent homology. Python bindings were modified
    for performance.

    Persistence diagrams produced by this class must be interpreted with
    care due to the presence of padding triples which carry no information.
    See :meth:`transform` for additional information.

    References
    ----------
    [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference \
        Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent_\
        cohomology.html>`_.

    """

    _hyperparameters = {
        'homology_dimensions': {
            'type': (list, tuple), 'of': {
                'type': int, 'in': Interval(0, np.inf, closed='left')}},
        'coeff': {'type': int, 'in': Interval(2, np.inf, closed='left')},
        'max_edge_length': {
            'type': Real, 'in': Interval(0, np.inf, closed='right')},
        'infinity_values': {
            'type': (Real, type(None)),
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
            Xdgms[0] = Xdgms[0][1:, :]  # Remove one infinite bar

        # Add dimension as the third elements of each (b, d) tuple
        Xdgms = {dim: np.hstack([Xdgms[dim],
                                 dim * np.ones((Xdgms[dim].shape[0], 1),
                                               dtype=Xdgms[dim].dtype)])
                 for dim in self._homology_dimensions}
        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list
            Input data representing a collection of point clouds. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays. The rows of
            elements in `X` are interpreted as vectors in Euclidean space and.
            and, when `X` is a list, warnings are issued when the number of
            columns (dimension of the Euclidean space) differs among samples.

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
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data representing a collection of point clouds. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays. The rows of
            elements in `X` are interpreted as vectors in Euclidean space and.
            and, when `X` is a list, warnings are issued when the number of
            columns (dimension of the Euclidean space) differs among samples.

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
    def plot(Xt, sample=0, homology_dimensions=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions)
