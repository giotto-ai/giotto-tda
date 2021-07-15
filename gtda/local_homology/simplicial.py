from numbers import Real
from types import FunctionType

import numpy as np
import warnings

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from gtda.homology import VietorisRipsPersistence
from gtda.utils.intervals import Interval
from gtda.utils.validation import validate_params
from gtda.plotting import plot_diagram
from gtda.base import PlotterMixin
from gtda.utils._docs import adapt_fit_transform_docs


class LocalVietorisRipsBase(BaseEstimator,
                            TransformerMixin,
                            PlotterMixin):
    """
    Base class for KNeighboursLocalVietorisRips and RadiusLocalVietorisRips.
    LocalVietorisRipsBase is not meant to be used. Please see documentation
    for KNeighboursLocalVietorisRips and RadiusLocalVietorisRips."""

    def __init__(self, metric='euclidean', homology_dimensions=(1, 2),
                 neighborhoud_param=(1, 2), n_jobs=1):
        """
        Initializes the base class by setting the parameters, initializes the
        object used for computing persistence homology, and checking that the
        parameters were initialized correctly."""

        # metric for the point cloud
        self.metric = metric

        # topological dimension of features to be computed
        self.homology_dimensions = homology_dimensions

        self.n_jobs = n_jobs

        # Object is used to compute persistence diagrams
        self.homology = VietorisRipsPersistence(
                            metric='precomputed',
                            collapse_edges=True,
                            homology_dimensions=self.homology_dimensions,
                            n_jobs=self.n_jobs)

        # tuple of parameters defining 'neighborhouds' of points. These
        # parameters are input in the Transformer objects determining what
        # points lie in the 'neighborhouds' of each points. The points outside
        # the 'neighborhoud' defined by the largest entry are discarded, and
        # the points between the smaller and largest 'neighborhouds' are 'coned
        # off'. See more in the corresponding fit methods.
        self.neighborhoud_param = neighborhoud_param

        # make sure the neighborhoud_param has been set correctly.
        if self.neighborhoud_param[0] > self.neighborhoud_param[1]:
            warnings.warn('First neighborhoud_param should be smaller than second.\
                The values are permuted')
            self.neighborhoud_param = (self.neighborhoud_param[1],
                                       self.neighborhoud_param[0])
        if self.neighborhoud_param[1] == 0:
            warnings.warn('Second neighborhoud_param has to be strictly greater than 0.\
                Second radius set to 1.')
            self.radii = (self.radii[0], 1)
        if self.neighborhoud_param[0] == self.neighborhoud_param[1]:
            warnings.warn('For meaningfull features, the first neighborhoud_param should\
                be strictly smaller than second.')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Computes the local persistence diagrams at each element of X, and
        returns a list of persistence diagrams, indexed as the points of X.
        This is done in several steps:
            - First compute the nearest neighbors in the point cloud that
            was fitted on, for both values in n_neighbors.
            - For each point, compute the relevant points (corresponding to
            the larger neighborhoud_param value), the close points
            (corresponding to the smaller neighborhoud_param value), and the
            annulus to cone off (relevant points, but not close points).
            Compute the distance matrix of the relevant points, and add an
            additional row and column corresponding to the coning off point.
            - Finally compute the persistence diagrams of each coned matrices.

        Parameters
        ----------

        X : ndarray of shape (n_points, dimension)
             Input data representing  point cloud:
             an array of shape ``(n_points, n_dimensions)``.


        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays.
            ``n_features`` equals :math:`\\sum_q n_q`, where :math:`n_q`
            is the maximum number of topological features in dimension
            :math:`q` across all samples in `X`.

        """

        check_is_fitted(self)
        check_array(X, accept_sparse=False)

        # sparse binary matrices where rows indicate the indices of points
        # which are nearest neighbors to the row's index point.
        Xt_close = self.close_neighbors_.transform(X)
        Xt_relevant = self.relevant_neighbors_.transform(X)

        coned_mats = []
        for i in range(len(X)):
            # get indices of points close to point at index i
            close_indices = Xt_close.getrow(i).indices
            # get indices of points in second 'neighborhoud'
            relevant_indices = Xt_relevant.getrow(i).indices
            annulus_indices = list(set(relevant_indices) - set(close_indices))
            # Order them such that the last ones are the ones to cone off
            reordered_relevant_indices = np.concatenate((close_indices,
                                                         annulus_indices))
            if len(close_indices) == 0:
                # The coned off space retracts to the cone point
                coned_mat = np.zeros((1, 1))
            else:
                # Fetch the coordinates
                relevant_points = [self.relevant_neighbors_._fit_X[int(y)]
                                   for y in reordered_relevant_indices]
                # Dense distance matrix between all relevant points
                local_mat = squareform(pdist(relevant_points,
                                             metric=self.metric))
                # Now add the cone point:
                new_row = np.concatenate((np.ones(len(close_indices))*np.inf,
                                          np.zeros(len(annulus_indices))))
                new_col = np.concatenate((new_row, [0]))
                pre_cone = np.concatenate((local_mat, [new_row]))
                coned_mat = np.concatenate(
                                           (pre_cone, np.array([new_col],
                                                               dtype=float).T),
                                           axis=1)
            coned_mats += [coned_mat]
        # Compute the Vietoris Rips Persistence diagrams
        Xt = self.homology.fit_transform(coned_mats)
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
class KNeighborsLocalVietorisRips(LocalVietorisRipsBase):
    """
    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Eclidean space, or an abstract :ref:`metric space
    <finite_metric_spaces_and_point_clouds>` encoded by a distance matrix,
    information about the local topology around each point is summarized
    in a list of persistence diagrams. This is done by first isolating
    appropriate neighborhoods around each point, using a nearest neighbor
    transformer then 'coning' off an annulus around each point, and computing
    the correponding associated persistence diagram. The output can then be
    used to explore the point cloud, or fead into a vectorizer to obtain
    features.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        Input data is to be interpreted as a point cloud (i.e. feature arrays),
        and `metric`determines a rule with which to calculate distances between
        pairs of points (i.e. row vectors). If `metric` is a string, it must be
        one of the options allowed by :func:`scipy.spatial.distance.pdist`
        for its metric parameter, or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``'euclidean'``, ``'manhattan'`` or ``'cosine'``. If `metric` is a
        callable, it should take pairs of vectors (1D arrays) as input and, for
        each two vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    n_neighbors: tuple, optional, default: ``(10, 50)``, has to
        consist of two non-negative integers. This defines the number of points
        in the first and second neighborhoods considered.

    homology_dimensions: tuple, optional, default: ``(1, 2)``. Dimensions
        (non-negative integers) of the topological features to be detected.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'n_neighbors': {'type': (tuple, list),
                        'of': {type: int,
                               'in': Interval(1, np.inf, closed='left')}
                        },
        'homology_dimensions': {
            'type': (tuple, list),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            }
        }

    def __init__(self, metric='euclidean', homology_dimensions=(1, 2),
                 n_neighbors=(1, 2), n_jobs=1):
        self.n_neighbors = n_neighbors
        super().__init__(metric=metric,
                         homology_dimensions=homology_dimensions,
                         neighborhoud_param=self.n_neighbors,
                         n_jobs=n_jobs)

    def fit(self, X, y=None):
        """
        Initiates and fits the transformers that efficiently computes the
        nearest neighbors of given points.
        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------

        X : ndarray of shape (n_points, dimension)
            Input data representing  point cloud. Can be either
            a point cloud: an array of shape ``(n_points, n_dimensions)``.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])
        check_array(X, accept_sparse=False)

        # make sure that the parameters are set correctly
        self.size_ = len(X)
        if self.size_ <= self.neighborhoud_param[0]:
            warnings.warn('First n_neighbors is too large to be relevant.\
                             Consider reducing it.')
            self.neighborhoud_param = (self.size_, self.size_)
        if self.size_ < self.neighborhoud_param[1]:
            warnings.warn('Second n_neighbors is too large to be relevant.\
                             Consider reducing it.')
            self.neighborhoud_param = (self.neighborhoud_param[0], self.size_)

        # Objects used for finding nearest neighbors
        self.close_neighbors_ = KNeighborsTransformer(
                                    mode='connectivity',
                                    n_neighbors=self.neighborhoud_param[0],
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)

        self.relevant_neighbors_ = KNeighborsTransformer(
                                    mode='connectivity',
                                    n_neighbors=self.neighborhoud_param[1],
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)

        self.close_neighbors_.fit(X)
        self.relevant_neighbors_.fit(X)
        return self


@adapt_fit_transform_docs
class RadiusLocalVietorisRips(LocalVietorisRipsBase):
    """
    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Eclidean space, or an abstract :ref:`metric space
    <finite_metric_spaces_and_point_clouds>` encoded by a distance matrix,
    information about the local topology around each point is summarized
    in a list of persistence diagrams. This is done by first isolating
    appropriate neighborhoods around each point, using a nearest neighbor
    transformer then 'coning' off points in an annulus around each point,
    and computing correponding associated persistence diagram. The output
    can then be used to explore the point cloud, or fead into a vectorizer
    to obtain features.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        Input data is to be interpreted as a point cloud (i.e. feature arrays),
        and `metric`determines a rule with which to calculate distances between
        pairs of points (i.e. row vectors). If `metric` is a string, it must be
        one of the options allowed by :func:`scipy.spatial.distance.pdist` for
        its `metric` parameter, or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``'euclidean'``, ``'manhattan'`` or ``'cosine'``. If `metric` is a
        callable, it should take pairs of vectors (1D arrays) as input and, for
        each two vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    radii: tuple, optional, default: ``(0.0, 1.0)`` has to consist of two
    non-negative floats. This determines the radius of the first and second
    neighborhood around points considered.

    homology_dimensions: tuple, optional, default: ``(1, 2)``. Dimensions
        (non-negative integers) of the topological features to be detected.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'radii': {'type': (tuple, list),
                  'of': {type: Real, 'in': Interval(0, np.inf, closed='left')}
                  },
        'homology_dimensions': {
            'type': (tuple, list),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            }
        }

    def __init__(self, metric='euclidean', homology_dimensions=(1, 2),
                 radii=(1, 2), n_jobs=1):
        self.radii = radii
        super().__init__(metric=metric,
                         homology_dimensions=homology_dimensions,
                         neighborhoud_param=self.radii,
                         n_jobs=n_jobs)

    def fit(self, X, y=None):
        """
        Initiates and fits the transformers that efficiently computes the
        nearest neighbors of given points.
        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------

        X : ndarray of shape (n_points, dimension)
            Input data representing  point cloud. Can be either
            a point cloud: an array of shape ``(n_points, n_dimensions)``.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])
        check_array(X, accept_sparse=False)

        # Objects used for finding nearest neighbors
        self.close_neighbors_ = RadiusNeighborsTransformer(
                                    mode='connectivity',
                                    radius=self.neighborhoud_param[0],
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)

        self.relevant_neighbors_ = RadiusNeighborsTransformer(
                                    mode='connectivity',
                                    radius=self.neighborhoud_param[1],
                                    metric=self.metric,
                                    n_jobs=self.n_jobs)

        self.close_neighbors_.fit(X)
        self.relevant_neighbors_.fit(X)
        return self
