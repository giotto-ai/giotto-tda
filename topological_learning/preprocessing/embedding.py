import numpy as np
import pandas as pd
import sklearn as sk

from joblib import Parallel, delayed

from pandas.core.resample import Resampler as rsp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
import math as m


class TakensEmbedder(BaseEstimator, TransformerMixin):
    """
    Transformer that return a time serie embedded according to Taken's sliding window.

    Parameters
    ----------
    nPeriodsList : list of int
        List of number of periods

    outerWindowDuration: pd.time
        Duration of the outer sliding window

    outerWindowStride: pd.time
        Stride of the outer sliding window

    innerWindowDuration: pd.time
        Duration of the inner sliding window

    innerWindowStride: pd.time
        Stride of the inner sliding window

    Attributes
    ----------
    inputShape : tuple The shape the data passed to :meth:`fit`
    """

    implemented_embedding_parameters_types = ['fixed', 'search']

    def __init__(self, outer_window_duration=20, outer_window_stride=2, embedding_parameters_type='search',
                 embedding_dimension=5, embedding_time_delay=1, embedding_stride=1, n_jobs=1):
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

        self.is_fitted = True
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
        check_is_fitted(self, ['is_fitted'])

        if X.shape[0] < self.outer_window_duration:
            raise ValueError('Not enough data to have a single outer window.')

        X_transformed = self._embed(X, self.outer_window_duration, self.outer_window_stride,
                                   self.embedding_time_delay, self.embedding_dimension, self.embedding_stride)
        return X_transformed
