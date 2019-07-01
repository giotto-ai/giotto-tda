import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils._joblib import Parallel, delayed

import numpy as np
import gudhi as gd
from ripser import ripser


class VietorisRipsPersistence(BaseEstimator, TransformerMixin):
    """
    Transformer for the calculation of persistence diagrams from Vietoris-Rips filtration.

    Parameters
    ----------
    samplingType : str
        The type of sampling

        - data_type: string, must equal either 'points' or 'distance_matrix'.
        - data_iter: an iterator. If data_iter is 'points' then each object in the iterator
          should be a numpy array of dimension (number of points, number of coordinates),
          or equivalent nested list structure. If data_iter is 'distance_matrix' then each
          object in the iterator should be a full (symmetric) square matrix (numpy array)
          of shape (number of points, number of points), __or a sparse distance matrix

    Attributes
    ----------
    isFitted : boolean
        Whether the transformer has been fitted
    """

    def __init__(self, data_type='points', max_edge_length=np.inf, homology_dimensions=[0, 1], pad=True, n_jobs=1):
        self.data_type = data_type
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.pad = pad
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'data_type': self.data_type,
                'max_edge_length': self.max_edge_length,
                'homology_dimensions': self.homology_dimensions,
                'pad': self.pad,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(data_type):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        implemented_data_types = ['points', 'distance_matrix']

        if data_type not in implemented_data_types:
            return ValueError('The data type you specified is not implemented')

    def _ripser_diagram(self, X, is_distance_matrix):
        diagram = ripser(X, distance_matrix = is_distance_matrix, maxdim = max(self.homology_dimensions), thresh = self.max_edge_length)['dgms']

        if 0 in self.homology_dimensions:
            diagram[0] = diagram[0][:-1, :]

        return { dimension: diagram[dimension] for dimension in self.homology_dimensions }

    def _pad_diagram(self, diagram, max_length_list):
        padList = [ ((0, max(0, max_length_list[i] - diagram[dimension].shape[0])), (0,0)) for i, dimension in enumerate(self.homology_dimensions) ]
        return { dimension: np.pad(diagram[dimension], padList[i], 'constant') for i, dimension in enumerate(self.homology_dimensions) }

    def _stack_padded_diagrams(self, diagrams):
        stacked_diagrams = { dimension: np.stack([diagrams[i][dimension] for i in range(len(diagrams))], axis=0) for dimension in self.homology_dimensions }

        # for dimension in self.homology_dimensions:
        #     if stackedDiagrams[dimension].size == 0:
        #         del stackedDiagrams[dimension]
        return stacked_diagrams

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
        self._validate_params(self.data_type)

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

        is_distance_matrix = (self.data_type == 'distance_matrix')

        X_transformed = Parallel(n_jobs = self.n_jobs) ( delayed(self._ripser_diagram)(X[i, :, :], is_distance_matrix)
                                                        for i in range(X.shape[0]) )

        if self.pad:
            max_length_list = [ np.max([ X_transformed[i][dimension].shape[0] for i in range(len(X_transformed)) ]) for dimension in self.homology_dimensions ]
            X_transformed = Parallel(n_jobs = self.n_jobs) ( delayed(self._pad_diagram)(X_transformed[i], max_length_list)
                                                            for i in range(len(X_transformed)) )
            X_transformed = self._stack_padded_diagrams(X_transformed)

        return X_transformed


class PersistentEntropy(BaseEstimator, TransformerMixin):
    def __init__(self, len_vector=8, n_jobs=1):
        self.len_vector = len_vector
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'len_vector': self.len_vector, 'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params():
        """A class method that checks whether the hyperparameters and the input parameters
            of the :meth:'fit' are valid.
            """
        pass

    def _persistent_entropy(self, X):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_normalized = X_lifespan / np.sum(X_lifespan, axis=1).reshape((-1, 1))
        return - np.sum(np.nan_to_num(X_normalized * np.log(X_normalized)), axis=1).reshape((-1, 1))

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
        self._validate_params()

        self.is_fitted = True
        return self

    #@jit
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

        n_samples = X[next(iter(X.keys()))].shape[0]
        n_dimensions = len(X.keys())

        slice_indices = list(range(0, n_samples, self.len_vector)) + [n_samples]
        n_slices = len(slice_indices) - 1

        X_transformed = Parallel(n_jobs=self.n_jobs) ( delayed(self._persistent_entropy) (X[dimension][slice_indices[i] : slice_indices[i+1]])
                                                       for dimension in X.keys() for i in range(n_slices) )

        X_transformed = np.hstack( [ np.concatenate([X_transformed[i*n_slices + j] for j in range(n_slices)], axis=0)  for i in range(n_dimensions) ])

        return X_transformed
