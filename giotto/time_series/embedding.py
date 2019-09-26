# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
#          Umberto Lupo <u.lupo@l2f.ch>
# License: TBD

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted


class TakensEmbedder(BaseEstimator, TransformerMixin):
    r"""Transformer returning a representation of a scalar-valued time series as
    a time series of point clouds.

    Based on the following time-delay embedding
    technique named after `F. Takens <https://doi.org/10.1007/BFb0091924>`_:
    given a time series :math:`X(t)`, one extracts a set of vectors in
    :math:`\mathbb{R}^d`, each of the form
        :math:`\Xi_i := [X(t_i), X(t_i + \tau), ..., X(t_i + (d-1)\tau)]`.
    The set :math:`\{\Xi_i\}` is called the Takens embedding of the time
    series, :math:`\tau` is called the embedding time delay, :math:`d` is
    called the embedding dimension, and the difference between :math:`t_i` and
    :math:`t_{i-1}` is called the embedding stride.

    If :math:`d` and :math:`\tau` are not explicitly set by the user, suitable
    values are calculated during ``fit()``.

    During ``transform()``, a Takens embedding procedure is applied on
    intervals of the input time series called "outer windows",
    in a sliding-window fashion. This allows to track the evolution of the
    dynamics underlying the time series.

    Parameters
    ----------
    outer_window_duration: int, default: 20
        Duration of the outer sliding window.

    outer_window_stride: int, default: 2
        Stride of the outer sliding window.

    embedding_parameters_type: 'search' | 'fixed', default: 'search'
        If set to 'fixed' and if values for ``embedding_time_delay`` and
        ``embedding_dimension`` are provided, these values are used in
        ``transform()``.
        If set to 'search' and if ``embedding_time_delay`` and
        ``embedding_dimension`` are not set, optimal values are
        automatically found for those parameters using mutual information
        (``embedding_time_delay``) and false nearest neighbors (
        ``embedding_dimension``) criteria.
        If set to 'search' and if ``embedding_time_delay`` and
        ``embedding_dimension`` are set, a similar optimization is carried
        out, but the final values are constrained to be not greater than the
        values initially set.

    embedding_time_delay: int, default: 1
        Time delay between two consecutive values for constructing one
        embedded point. If ``embedding_parameters_type`` is 'search',
        it corresponds to the maximal embedding time delay that will be
        considered.

    embedding_dimension: int, default: 5
        Dimension of the embedding space. If ``embedding_parameters_type`` is
        'search', it corresponds to the maximum embedding dimension that will
        be considered.

    embedding_stride: int, default: 1
        Stride duration between two consecutive embedded points. It defaults
        to 1 as this is the usual value in the statement of Takens's embedding
        theorem.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    embedding_time_delay_: int
        Actual embedding time delay used to embed. If
        ``embedding_parameters_type`` is 'search', it is the calculated
        optimal embedding time delay. Otherwise it has the same value as
        ``embedding_time_delay``.

    embedding_dimension_: int
        Actual embedding dimension used to embed. If
        ``embedding_parameters_type`` is 'search', it is the calculated
        optimal embedding dimension. Otherwise it has the same value as
        ``embedding_dimension``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from giotto.time_series import TakensEmbedder
    >>> # Create a noisy signal sampled
    >>> signal_noise = np.asarray([np.sin(x /40) - 0.5 + np.random.random()
    ...     for x in range(0,1000)])
    >>> # Set up the Takens Embedder
    >>> outer_window_duration = 50
    >>> outer_window_stride = 5
    >>> embedder = TakensEmbedder(
    >>>     outer_window_duration=outer_window_duration,
    ...     outer_window_stride=outer_window_stride,
    ...     embedding_parameters_type='search',
    ...     embedding_dimension=5,
    ...     embedding_time_delay=1, n_jobs=-1)
    >>> # Fit and transform the DataFrame
    >>> embedder.fit(signal_noise)
    >>> embedded_noise = embedder.transform(signal_noise)
    >>> print('Optimal embedding time delay based on mutual information:',
    ...       embedder.embedding_time_delay_)
    Optimal embedding time delay based on mutual information: 1
    >>> print('Optimal embedding dimension based on false nearest neighbors:',
    ...       embedder.embedding_dimension_)
    Optimal embedding dimension based on false nearest neighbors: 3

    """

    def __init__(self, outer_window_duration=20, outer_window_stride=2,
                 embedding_parameters_type='search',
                 embedding_time_delay=1, embedding_dimension=5,
                 embedding_stride=1, n_jobs=None):
        self.outer_window_duration = outer_window_duration
        self.outer_window_stride = outer_window_stride
        self.embedding_parameters_type = embedding_parameters_type
        self.embedding_time_delay = embedding_time_delay
        self.embedding_dimension = embedding_dimension
        self.embedding_stride = embedding_stride
        self.n_jobs = n_jobs

    def _validate_params(self, X):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.
        """
        implemented_embedding_parameters_types = ['fixed', 'search']

        if self.embedding_parameters_type not in \
                implemented_embedding_parameters_types:
            raise ValueError(
                'The embedding parameters type %s is not supported' %
                self.embedding_parameters_type)

        if X.shape[0] < self.outer_window_duration:
            raise ValueError('Not enough data to have a single outer window.')

    @staticmethod
    def _embed(X, outer_window_duration, outer_window_stride,
               embedding_time_delay, embedding_dimension, embedding_stride=1):
        n_outer_windows = \
            (X.shape[0] - outer_window_duration) // outer_window_stride + 1
        n_points = (outer_window_duration - embedding_time_delay *
                    (embedding_dimension - 1) - 1) // embedding_stride + 1

        X = np.flip(X)

        XEmbedded = np.stack([
            np.stack([
                X[i * outer_window_stride + j * embedding_stride:
                  i * outer_window_stride + j * embedding_stride +
                  embedding_time_delay * (embedding_dimension - 1) + 1:
                  embedding_time_delay].flatten()
                for j in range(0, n_points)])
            for i in range(0, n_outer_windows)])

        return np.flip(XEmbedded).reshape(
            (n_outer_windows, n_points, embedding_dimension))

    @staticmethod
    def _mutual_information(X, embedding_time_delay, n_bins):
        """This function calculates the mutual information given the delay
        """
        contingency = np.histogram2d(X.reshape((-1,))[:-embedding_time_delay],
                                     X.reshape((-1,))[embedding_time_delay:],
                                     bins=n_bins)[0]
        mutual_information = mutual_info_score(None, None,
                                               contingency=contingency)
        return mutual_information

    @staticmethod
    def _false_nearest_neighbors(X, embedding_time_delay, embedding_dimension,
                                 embedding_stride=1):
        """Calculates the number of false nearest neighbours of embedding
        dimension"""
        XEmbedded = TakensEmbedder._embed(X, X.shape[0], 1,
                                          embedding_time_delay,
                                          embedding_dimension,
                                          embedding_stride)
        XEmbedded = XEmbedded.reshape((XEmbedded.shape[1], XEmbedded.shape[2]))

        neighbor = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(
            XEmbedded)
        distances, indices = neighbor.kneighbors(XEmbedded)
        distance = distances[:, 1]
        XNeighbor = X[indices[:, 1]]

        epsilon = 2.0 * np.std(X)
        tolerance = 10

        dim_by_delay = -embedding_dimension * embedding_time_delay
        non_zero_distance = distance[:dim_by_delay] > 0

        false_neighbor_criteria = \
            np.abs(np.roll(X, dim_by_delay)[
                   X.shape[0] - XEmbedded.shape[0]:dim_by_delay] -
                   np.roll(XNeighbor, dim_by_delay)[:dim_by_delay]) \
            / distance[:dim_by_delay] > tolerance

        limited_dataset_criteria = distance[:dim_by_delay] < epsilon

        n_false_neighbors = np.sum(
            non_zero_distance * false_neighbor_criteria *
            limited_dataset_criteria)
        return n_false_neighbors

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
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
            Returns self.
        """
        self._validate_params(X)

        if self.embedding_parameters_type == 'search':
            mutual_information_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._mutual_information)(X, embedding_time_delay,
                                                  n_bins=100)
                for embedding_time_delay in
                range(1, self.embedding_time_delay + 1))
            self.embedding_time_delay_ = mutual_information_list.index(
                min(mutual_information_list)) + 1

            n_false_nbhrs_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._false_nearest_neighbors)(
                    X, self.embedding_time_delay, embedding_dimension,
                    embedding_stride=1) for embedding_dimension in
                range(1, self.embedding_dimension + 3))

            variation_list = [np.abs(n_false_nbhrs_list[emb_dim - 1] - 2 *
                n_false_nbhrs_list[emb_dim] + n_false_nbhrs_list[emb_dim + 1])
                / (n_false_nbhrs_list[emb_dim] + 1) / emb_dim
                for emb_dim in range(2, self.embedding_dimension + 1)]

            self.embedding_dimension_ = e_d_temp + 2

        else:
            self.embedding_time_delay_ = self.embedding_time_delay
            self.embedding_dimension_ = self.embedding_dimension

        self._is_fitted = True
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
        X_transformed : ndarray, shape (n_outer_windows, n_points,
        embedding_dimension_)
            Array of embedded point cloud per outer window.
            ``n_outer_windows`` is  ``(n_samples - outer_window_duration) //
            outer_window_stride + 1``, and ``n_points`` is ``(
            outer_window_duration - embedding_time_delay *
            embedding_dimension) // embedding_stride + 1``.

        """

        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = self._embed(X, self.outer_window_duration,
                                    self.outer_window_stride,
                                    self.embedding_time_delay_,
                                    self.embedding_dimension_,
                                    self.embedding_stride)
        return X_transformed
