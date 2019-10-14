"""Time series embedding."""
# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator
from ..base import TransformerResamplerMixin
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, check_array, column_or_1d
from ..utils.validation import validate_params


class SlidingWindow(BaseEstimator, TransformerResamplerMixin):
    """Concatenates results of multiple transformer objects.
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters
    ----------
    width : int, default: ``1``
        Width of the sliding window.

    stride : int, default: ``1``
        Stride of the sliding window.

    Examples
    --------
    >>> from giotto.time_series import SlidingWindow
    """
    _hyperparameters = {'width': [int, (1, np.inf)],
                        'stride': [int, (1, np.inf)]}

    def __init__(self, width=1, stride=1):
        self.width = width
        self.stride = stride

    def _slice_windows(self, X):
        n_windows = (X.shape[0] - self.width) // self.stride + 1

        window_slices = [(i * self.stride, self.width + i * self.stride)
                         for i in range(n_windows)]
        return window_slices

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self

        """
        validate_params(self.get_params(), self._hyperparameters)
        check_array(X, ensure_2d=False, allow_nd=True)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Slide windows over X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, [n_features, ])
            Input data.

        y : None
            Ignored.

        Returns
        -------
        Xt : ndarray, shape (n_windows, n_samples_window, \
             n_features)

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        X = check_array(X, ensure_2d=False, allow_nd=True)

        window_slices = self._slice_windows(X)

        Xt = np.stack([X[begin:end] for begin, end in window_slices])
        return Xt

    def resample(self, y, X=None):
        """Resample y.

        Parameters
        ----------
        y : ndarray, shape (n_samples, n_features)
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target.
            ``n_samples_new = n_samples - 1``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        y = column_or_1d(y)

        yt = y[self.width - 1:: self.stride]
        return yt


class TakensEmbedding(BaseEstimator, TransformerResamplerMixin):
    """Representation of a univariate time series as a time series of
    point clouds.

    Based on the following time-delay embedding technique named after F.
    Takens [1]_: given a time series :math:`X_t`, one extracts a list of
    vectors in :math:`\\mathbb{R}^d`, each of the form
    :math:`\\mathcal{X}_i := (X_{t_i}, X_{t_i + \\tau}, \\ldots , X_{t_i + (
    d-1)\\tau})`. The set :math:`\\{\\mathcal{X}_i\\}_i` is called the `Takens
    embedding <LINK TO GLOSSARY>`_ of the time series, :math:`\\tau` is
    called the embedding time delay, :math:`d` is called the embedding
    dimension, and the difference between :math:`t_i` and :math:`t_{i-1}` is
    called the embedding stride.

    If :math:`d` and :math:`\\tau` are not explicitly set, suitable values
    are calculated during :meth:`fit`. [2]_

    Parameters
    ----------
    parameters_type : ``'search'`` | ``'fixed'``, default: ``'search'``
        If set to ``'fixed'`` and if values for `embedding_time_delay` and
        `dimension` are provided, these values are used in :meth:`transform`.
        If set to ``'search'`` and if `embedding_time_delay` and `dimension`
        are not set, optimal values are automatically found for those
        parameters using criteria based on mutual information (`time_delay`)
        and false nearest neighbors. [2]_
        If set to 'search' and if `time_delay` and `dimension` are set,
        a similar optimization is carried out, but the final values are
        constrained to be not greater than the values initially set.

    time_delay : int, default: ``1``
        Time delay between two consecutive values for constructing one
        embedded point. If `parameters_type` is ``'search'``,
        it corresponds to the maximal embedding time delay that will be
        considered.

    dimension : int, default: ``5``
        Dimension of the embedding space. If `parameters_type` is ``'search'``,
        it corresponds to the maximum embedding dimension that will be
        considered.

    stride : int, default: ``1``
        Stride duration between two consecutive embedded points. It defaults
        to 1 as this is the usual value in the statement of Takens's embedding
        theorem.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    time_delay_ : int
        Actual embedding time delay used to embed. If
        `parameters_type` is ``'search'``, it is the calculated optimal
        embedding time delay. Otherwise it has the same value as `time_delay`.

    dimension_ : int
        Actual embedding dimension used to embed. If `parameters_type` is
        ``'search'``, it is the calculated optimal embedding dimension.
        Otherwise it has the same value as `dimension`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from giotto.time_series import TakensEmbedding
    >>> # Create a noisy signal sampled
    >>> signal_noise = np.asarray([np.sin(x /40) - 0.5 + np.random.random()
    ...     for x in range(0, 1000)])
    >>> # Set up the transformer
    >>> outer_window_duration = 50
    >>> outer_window_stride = 5
    >>> embedder = TakensEmbedding(
    >>>     outer_window_duration=outer_window_duration,
    ...     outer_window_stride=outer_window_stride,
    ...     parameters_type='search',
    ...     dimension=5,
    ...     time_delay=1, n_jobs=-1)
    >>> # Fit and transform the DataFrame
    >>> embedder.fit(signal_noise)
    >>> embedded_noise = embedder.transform(signal_noise)
    >>> print('Optimal embedding time delay based on mutual information:',
    ...       embedder.time_delay_)
    Optimal embedding time delay based on mutual information: 1
    >>> print('Optimal embedding dimension based on false nearest neighbors:',
    ...       embedder.dimension_)
    Optimal embedding dimension based on false nearest neighbors: 3

    See also
    --------
    giotto.homology.VietorisRipsPersistence

    References
    ----------
    .. [1] F. Takens, "Detecting strange attractors in turbulence". In: Rand
           D., Young LS. (eds) *Dynamical Systems and Turbulence, Warwick
           1980*. Lecture Notes in Mathematics, vol 898. Springer, 1981;
           doi: `10.1007/BFb0091924 <https://doi.org/10.1007/BFb0091924>`_.

    .. [2] N. Sanderson, "Topological Data Analysis of Time Series using
           Witness Complexes", PhD thesis, University of Colorado at
           Boulder, 2018; `https://scholar.colorado.edu/math_gradetds/67
           <https://scholar.colorado.edu/math_gradetds/67>`_.

    """
    _hyperparameters = {'parameters_type': [str, ['fixed', 'search']],
                        'time_delay': [int, (1, np.inf)],
                        'dimension': [int, (1, np.inf)],
                        'stride': [int, (1, np.inf)]}

    def __init__(self, parameters_type='search', time_delay=1, dimension=5,
                 stride=1, n_jobs=None):
        self.parameters_type = parameters_type
        self.time_delay = time_delay
        self.dimension = dimension
        self.stride = stride
        self.n_jobs = n_jobs

    @staticmethod
    def _embed(X, time_delay, dimension, stride):
        n_points = (X.shape[0] - time_delay * dimension) // stride + 1

        X = np.flip(X)
        points_ = [X[j * stride:
                     j * stride + time_delay * dimension:
                     time_delay].flatten() for j in range(0, n_points)]
        X_embedded = np.stack(points_)

        return np.flip(X_embedded).reshape((n_points, dimension))

    @staticmethod
    def _mutual_information(X, time_delay, n_bins):
        """Calculate the mutual information given the delay."""
        contingency = np.histogram2d(X.reshape((-1,))[:-time_delay],
                                     X.reshape((-1,))[time_delay:],
                                     bins=n_bins)[0]
        mutual_information = mutual_info_score(None, None,
                                               contingency=contingency)
        return mutual_information

    @staticmethod
    def _false_nearest_neighbors(X, time_delay, dimension,
                                 stride=1):
        """Calculate the number of false nearest neighbours of embedding
        dimension. """
        X_embedded = TakensEmbedding._embed(X, time_delay, dimension, stride)

        neighbor = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(
            X_embedded)
        distances, indices = neighbor.kneighbors(X_embedded)
        distance = distances[:, 1]
        XNeighbor = X[indices[:, 1]]

        epsilon = 2.0 * np.std(X)
        tolerance = 10

        dim_by_delay = -dimension * time_delay
        non_zero_distance = distance[:dim_by_delay] > 0

        false_neighbor_criteria = \
            np.abs(np.roll(X, dim_by_delay)[
                   X.shape[0] - X_embedded.shape[0]:dim_by_delay] -
                   np.roll(XNeighbor, dim_by_delay)[:dim_by_delay]) \
            / distance[:dim_by_delay] > tolerance

        limited_dataset_criteria = distance[:dim_by_delay] < epsilon

        n_false_neighbors = np.sum(
            non_zero_distance * false_neighbor_criteria *
            limited_dataset_criteria)
        return n_false_neighbors

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, 1)
            Input data.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        X = check_array(X.reshape(X.shape[0], 1), allow_nd=True)

        if self.parameters_type == 'search':
            mutual_information_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._mutual_information)(X, time_delay,
                                                  n_bins=100)
                for time_delay in
                range(1, self.time_delay + 1))
            self.time_delay_ = mutual_information_list.index(
                min(mutual_information_list)) + 1

            n_false_nbhrs_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._false_nearest_neighbors)(
                    X, self.time_delay, dim,
                    stride=1) for dim in
                range(1, self.dimension + 3))

            variation_list = [np.abs(n_false_nbhrs_list[dim - 1]
                                     - 2 * n_false_nbhrs_list[dim] +
                                     n_false_nbhrs_list[dim + 1])
                              / (n_false_nbhrs_list[dim] + 1) / dim
                              for dim in range(2, self.dimension + 1)]

            self.dimension_ = variation_list.index(min(variation_list)) + 2

        else:
            self.time_delay_ = self.time_delay
            self.dimension_ = self.dimension

        return self

    def transform(self, X, y=None):
        """Computes the embedding of X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, 1)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        Xt : ndarray, shape (n_points, n_dimension)
            Array of embedded point cloud per outer window.
            ``n_outer_windows = (n_samples - outer_window_duration) //
            outer_window_stride + 1``, and ``n_points = (
            outer_window_duration - time_delay * dimension) // stride + 1``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['time_delay_', 'dimension_'])
        X = check_array(X.reshape(X.shape[0], 1), allow_nd=True)

        Xt = self._embed(X, self.time_delay_, self.dimension_, self.stride)
        return Xt

    def resample(self, y, X=None):
        """Resample y.

        Parameters
        ----------
        y : ndarray, shape (n_samples, n_features)
            Target.

        X : None
            There is no need of input data,
            yet the pipeline API requires this parameter.

        Returns
        -------
        yt : ndarray, shape (n_samples_new, 1)
            The resampled target. ``n_samples_new = n_samples - 1``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['time_delay_', 'dimension_'])
        yt = column_or_1d(y)

        yt = y[self.time_delay_ * self.dimension_ - 1:: self.stride]
        return yt
