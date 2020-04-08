"""Persistent homology on grids."""
# License: GNU AGPLv3

from numbers import Real

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ._utils import _pad_diagram
from ..base import PlotterMixin
from ..externals.python import CubicalComplex, PeriodicCubicalComplex
from ..plotting import plot_diagram
from ..utils.intervals import Interval
from ..utils.validation import validate_params


class CubicalPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`filtered cubical complexes <cubical_complex>`.

    Given a :ref:`greyscale image <cubical_chains_and_cubical_homology>`,
    information about the appearance and disappearance of topological features
    (technically, :ref:`homology classes <homology_and_cohomology>`) of various
    dimensions and at different scales is summarised in the corresponding
    persistence diagram.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where
        :math:`p` equals `coeff`.

    periodic_dimensions : boolean ndarray of shape (n_dimensions,) or None, \
        optional, default: ``None``
        Periodicity of the boundaries along each of the axis, where
        ``n_dimensions`` is the dimension of the images of the collection. The
        boolean in the `d`th position expresses whether the boundaries along
        the `d`th axis are periodic. The default ``None`` is equivalent to
        passing ``numpy.zeros((n_dimensions,), dtype=np.bool)``, i.e. none of
        the boundaries are periodic.

    infinity_values : float or None, default : ``None``
        Which death value to assign to features which are still alive at
        filtration value `np.inf`. ``None`` assigns the maximum pixel
        values within all images passed to :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    periodic_dimensions_ : boolean ndarray of shape (n_dimensions,)
       Effective periodicity of the boundaries along each of the axis.
       Set in :meth:`fit`.

    infinity_values_ : float
       Effective death value to assign to features which have infinite
       persistence. Set in :meth:`fit`.

    See also
    --------
    VietorisRipsPersistence, SparseRipsPersistence, EuclideanCechPersistence

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing cubical persistent homology. Python bindings were modified
    for performance.

    Persistence diagrams produced by this class must be interpreted with
    care due to the presence of padding triples which carry no information.
    See :meth:`transform` for additional information.

    References
    ----------
    [1] P. Dlotko, "Cubical complex", 2015; `GUDHI User and Reference Manual \
    <http://gudhi.gforge.inria.fr/doc/latest/group__cubical__complex.html>`_.

    """

    _hyperparameters = {
        'homology_dimensions': {
            'type': (list, tuple), 'of': {
                'type': int, 'in': Interval(0, np.inf, closed='left')}},
        'coeff': {'type': int, 'in': Interval(2, np.inf, closed='left')},
        'periodic_dimensions': {
            'type': (np.ndarray, type(None)),
            'of': {'type': np.bool_}},
        'infinity_values': {'type': (Real, type(None))}}

    def __init__(self, homology_dimensions=(0, 1), coeff=2,
                 periodic_dimensions=None, infinity_values=None, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.periodic_dimensions = periodic_dimensions
        self.infinity_values = infinity_values
        self.n_jobs = n_jobs

    def _gudhi_diagram(self, X):
        cubical_complex = self._filtration(
            dimensions=X.shape,
            top_dimensional_cells=X.flatten(order="F"),
            **self._filtration_kwargs)
        Xdgms = cubical_complex.persistence(homology_coeff_field=self.coeff,
                                            min_persistence=0)

        # Separate diagrams by homology dimensions
        Xdgms = {dim: np.array([Xdgms[i][1] for i in range(len(Xdgms))
                                if Xdgms[i][0] == dim]).reshape((-1, 2))
                 for dim in self.homology_dimensions}

        # Add dimension as the third elements of each (b, d) tuple
        Xdgms = {dim: np.hstack([Xdgms[dim],
                                 dim * np.ones((Xdgms[dim].shape[0], 1),
                                               dtype=Xdgms[dim].dtype)])
                 for dim in self._homology_dimensions}
        return Xdgms

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_1, ..., n_pixels_d)
            Input data. Array of d-dimensional images.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_array(X, allow_nd=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        self._filtration_kwargs = {}
        if self.periodic_dimensions is None or \
           np.sum(self.periodic_dimensions) == 0:
            self._filtration = CubicalComplex
            self.periodic_dimensions_ = np.zeros(len(X) - 1, dtype=np.bool)
        else:
            self._filtration = PeriodicCubicalComplex
            self.periodic_dimensions_ = np.array(self.periodic_dimensions,
                                                 dtype=np.bool)
            self._filtration_kwargs['periodic_dimensions'] = \
                self.periodic_dimensions_

        if self.infinity_values is None:
            self.infinity_values_ = np.max(X)
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each image in `X`, compute the relevant persistence diagram
        as an array of triples [b, d, q]. Each triple represents a persistent
        topological feature in dimension q (belonging to `homology_dimensions`)
        which is born at b and dies at d. Only triples in which b < d are
        meaningful. Triples in which b and d are equal ("diagonal elements")
        may be artificially introduced during the computation for padding
        purposes, since the number of non-trivial persistent topological
        features is typically not constant across samples. They carry no
        information and hence should be effectively ignored by any further
        computation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_1, ..., n_pixels_d)
            Input data. Array of d-dimensional images.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

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
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._gudhi_diagram)(x) for x in Xt)

        max_n_points = {
            dim: max(1, np.max([x[dim].shape[0] for x in Xt])) for dim in
            self.homology_dimensions}
        min_values = {
            dim: min([np.min(x[dim][:, 0]) if x[dim].size else np.inf for x
                      in Xt]) for dim in self.homology_dimensions}
        min_values = {
            dim: min_values[dim] if min_values[dim] != np.inf else 0 for dim
            in self.homology_dimensions}
        Xt = Parallel(n_jobs=self.n_jobs)(delayed(_pad_diagram)(
            x, self._homology_dimensions, max_n_points, min_values)
            for x in Xt)
        Xt = np.stack(Xt)
        Xt = np.nan_to_num(Xt, posinf=self.infinity_values_)
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
