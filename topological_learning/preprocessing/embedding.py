import numpy as np
import pandas as pd
import sklearn as sk

from pandas.core.resample import Resampler as rsp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
import math as m


class TakensEmbedder(BaseEstimator, TransformerMixin):
    """
    Transformer that return a time serie embedded according to Taken's sliding window.
    In order to obtain meaningful topological features from a time series, we use a
    delayed-time embedding technique, invented by F. Takens in the late sixties. The
    idea is simple: given a time series X(t), one can extract a sequence of vectors
    of the form X_i := [(X(t_i)), X(t_i + 2 tau), ..., X(t_i + M tau)]. The difference
    between t_i and t_i-1 is called stride; the numbers M and tau are optimized authomatically
    in this example (they can be set by the user if needed).
    The outer window allows us to apply Takens embedding locally on a certain interval
    rather than over the whole time series. The result of this procedure is therefore a
    time series of point clouds with possibly interesting topologies.

    Parameters
    ----------

    outer_window_duration: int, default: 20
        Duration of the outer sliding window

    outer_window_stride: int, default: 2
        Stride of the outer sliding window

    embedding_dimension: int, default: 5
        Duration of the inner sliding window

    embedding_stride: int, default: 1
        Stride of the inner sliding window
        
    embedding_time_delay: int, default: 1
        The resampling time interval. The time-series will be resampled at
        t_n = t + n * embedding_time_delay for n in N.
    
    embedding_parameters_type: str, default: 'search'
        This parameter allow the user to choose the parameters ``embedding_tyme_delay``
        and ``embedding_stride`` manually or to optimize automatically optimize their values. It
        accepts two possible values:
    
        - 'search':
            It automatically optimize the parameters ``embedding_tyme_delay``
            and ``embedding_stride`` using mutual information and false nearest neightbors criteria.
        - 'fixed':
            It allows the user to choose the value for the parameters ``embedding_tyme_delay``
            and ``embedding_stride``.
            
    n_jobs: int or None, default: None
        Number of cores to be used in the computation.
            
    Attributes
    ----------
    inputShape : tuple The shape the data passed to :meth:`fit`
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import topological_learning.preprocessing as prep
    >>> import matplotlib.pyplot as plt
    >>> # Create a noisy signal sampled
    >>> signal_noise = np.asarray([np.sin(x /40) - 0.5 + np.random.random() for x in range(0,1000)])
    >>> # Set up the Takens Embedder
    >>> outerWindowDuration = 50
    >>> outerWindowStride = 5
    >>> embedder = prep.TakensEmbedder(outer_window_duration=outerWindowDuration,
    ... outer_window_stride=outerWindowStride,
    ... embedding_parameters_type='search', embedding_dimension=5, embedding_time_delay=1,
    ... n_jobs=-1)
    >>> # Fit and transform the DataFrame
    >>> embedder.fit(signal_noise)
    >>> zEmbedded = embedder.transform(signal_noise)
    >>> print('Optimal embedding time delay based on mutual information: ', embedder.embedding_time_delay)
    Optimal embedding time delay based on mutual information:  1
    >>> print('Optimal embedding dimension based on false nearest neighbors: ',
    ... embedder.embedding_dimension)
    Optimal embedding dimension based on false nearest neighbors:  3
    
    """

    implemented_embedding_parameters_types = ['fixed', 'search']

    def __init__(self, outer_window_duration=20, outer_window_stride=2, embedding_parameters_type='search',
                 embedding_dimension=5, embedding_time_delay=1, embedding_stride=1, n_jobs=None):
        self.outer_window_duration = outer_window_duration
        self.outer_window_stride = outer_window_stride
        self.embedding_parameters_type = embedding_parameters_type
        self.embedding_dimension = embedding_dimension
        self.embedding_time_delay = embedding_time_delay
        self.embedding_stride = embedding_stride
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'outer_window_duration': self.outer_window_duration, 'outer_window_stride': self.outer_window_stride,
                'embedding_parameters_type': self.embedding_parameters_type,
                'embedding_dimension': self.embedding_dimension,
                'embedding_time_delay': self.embedding_time_delay,
                'embedding_stride': self.embedding_stride,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(embedding_parameters_type):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        if embedding_parameters_type not in TakensEmbedder.implemented_embedding_parameters_types:
            raise ValueError('The embedding parameters type you specified is not implemented')

    @staticmethod
    def _embed(X, outer_window_duration, outer_window_stride, embedding_time_delay, embedding_dimension, embedding_stride=1):
        n_outer_windows = (X.shape[0] - outer_window_duration) // outer_window_stride + 1
        n_inner_windows = (outer_window_duration - embedding_time_delay*embedding_dimension) // embedding_stride + 1

        X = np.flip(X)

        XEmbedded = np.stack( [
            np.stack( [
                X[i*outer_window_stride + j*embedding_stride
                  : i*outer_window_stride + j*embedding_stride + embedding_time_delay*embedding_dimension
                  : embedding_time_delay].flatten()
                for j in range(0, n_inner_windows) ] )
            for i in range(0, n_outer_windows) ])

        return np.flip(XEmbedded).reshape((n_outer_windows, n_inner_windows, embedding_dimension))

    @staticmethod
    def _mutual_information(X, embedding_time_delay, numberBins):
        """This function calculates the mutual information given the delay
        """
        contingency = np.histogram2d(X[:-embedding_time_delay], X[embedding_time_delay:], bins=numberBins)[0]
        mutual_information = mutual_info_score(None, None, contingency=contingency)
        return mutual_information

    @staticmethod
    def _false_nearest_neighbors(X, embedding_time_delay, embedding_dimension, embedding_stride=1):
        """Calculates the number of false nearest neighbours of embedding dimension"""
        XEmbedded = TakensEmbedder._embed(X, X.shape[0], 1, embedding_time_delay, embedding_dimension, embedding_stride)
        XEmbedded = XEmbedded.reshape((XEmbedded.shape[1], XEmbedded.shape[2]))

        neighbor = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(XEmbedded)
        distances, indices = neighbor.kneighbors(XEmbedded)
        distance = distances[:, 1]
        XNeighbor = X[indices[:, 1]]

        epsilon = 2.0 * np.std(X)
        tolerance = 10

        non_zero_distance = distance[:-embedding_dimension*embedding_time_delay] > 0
        false_neighbor_criteria = np.abs(np.roll(X, -embedding_dimension*embedding_time_delay)[X.shape[0]-XEmbedded.shape[0]:-embedding_dimension*embedding_time_delay]
                                       - np.roll(XNeighbor, -embedding_dimension*embedding_time_delay)[:-embedding_dimension*embedding_time_delay]) \
                                       / distance[:-embedding_dimension*embedding_time_delay] > tolerance
        limited_dataset_criteria = distance[:-embedding_dimension*embedding_time_delay] < epsilon
        n_false_neighbors = np.sum(non_zero_distance * false_neighbor_criteria * limited_dataset_criteria)
        return n_false_neighbors

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params(self.embedding_parameters_type)

        if self.embedding_parameters_type == 'search':
            mutual_information_list = Parallel(n_jobs=self.n_jobs) ( delayed(self._mutual_information) (X, embedding_time_delay, numberBins=100)
                                                               for embedding_time_delay in range(1, self.embedding_time_delay+1) )
            self.embedding_time_delay = mutual_information_list.index(min(mutual_information_list)) + 1

            n_false_neighbors_list = Parallel(n_jobs=self.n_jobs) ( delayed(self._false_nearest_neighbors) (X, self.embedding_time_delay, embedding_dimension, embedding_stride=1)
                                                                 for embedding_dimension in range(1, self.embedding_dimension+3) )
            variation_list = [ np.abs(n_false_neighbors_list[embedding_dimension-1]-2*n_false_neighbors_list[embedding_dimension]+n_false_neighbors_list[embedding_dimension+1]) \
                              /(n_false_neighbors_list[embedding_dimension] + 1)/embedding_dimension for embedding_dimension in range(1, self.embedding_dimension+1) ]
            self.embedding_dimension = variation_list.index(min(variation_list)) + 1

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """ Implementation of the sk-learn transform function that samples the input.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        if X.shape[0] < self.outer_window_duration:
            raise ValueError('Not enough data to have a single outer window.')

        X_transformed = self._embed(X, self.outer_window_duration, self.outer_window_stride,
                                   self.embedding_time_delay, self.embedding_dimension, self.embedding_stride)
        return X_transformed
