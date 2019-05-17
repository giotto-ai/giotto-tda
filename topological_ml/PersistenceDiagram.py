import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from joblib import Parallel, delayed

import numpy as np
import gudhi as gd
from ripser import ripser

class VietorisRipsDiagram(BaseEstimator, TransformerMixin):
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

    def __init__(self, dataType='points', maxEdgeLength=np.inf, homologyDimensions=[0, 1], doPadding=True, n_jobs=1):
        self.dataType = dataType
        self.maxEdgeLength = maxEdgeLength
        self.homologyDimensions = homologyDimensions
        self.doPadding = doPadding
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {'dataType': self.dataType,
                'maxEdgeLength': self.maxEdgeLength,
                'homologyDimensions': self.homologyDimensions,
                'doPadding': self.doPadding,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(dataType):
        """A class method that checks whether the hyperparameters and the input parameters
           of the :meth:'fit' are valid.
        """
        implementedDataTypes = ['points', 'distanceMatrix']

        if dataType not in implementedDataTypes:
            return ValueError('The data type you specified is not implemented')

    def _ripser_diagram(self, X, isDistanceMatrix):
        diagram = ripser(X, distance_matrix = isDistanceMatrix, maxdim = max(self.homologyDimensions), thresh = self.maxEdgeLength)['dgms']

        if 0 in self.homologyDimensions:
            diagram[0] = diagram[0][:-1, :]

        return { dimension: diagram[dimension] for dimension in self.homologyDimensions }

    def _pad_diagram(self, diagram, maxLengthList):
        paddingList = [ ((0, max(0, maxLengthList[i] - diagram[dimension].shape[0])), (0,0)) for i, dimension in enumerate(self.homologyDimensions) ]
        return { dimension: np.pad(diagram[dimension], paddingList[i], 'constant') for i, dimension in enumerate(self.homologyDimensions) }

    def _stack_padded_diagrams(self, diagrams):
        stackedDiagrams = { dimension: np.stack([diagrams[i][dimension] for i in range(len(diagrams))], axis=0) for dimension in self.homologyDimensions }

        # for dimension in self.homologyDimensions:
        #     if stackedDiagrams[dimension].size == 0:
        #         del stackedDiagrams[dimension]
        return stackedDiagrams

    def fit(self, XList, y = None):
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
        self._validate_params(self.dataType)

        self.isFitted = True
        return self

    def transform(self, XList, y = None):
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
        check_is_fitted(self, ['isFitted'])

        XListTransformed = []

        isDistanceMatrix = (self.dataType == 'distanceMatrix')

        XTransformed = Parallel(n_jobs = self.n_jobs) ( delayed(self._ripser_diagram)(XList[0][i, :, :], isDistanceMatrix)
                                                        for i in range(XList[0].shape[0]) )

        if self.doPadding:
            maxLengthList = [ np.max([ XTransformed[i][dimension].shape[0] for i in range(len(XTransformed)) ]) for dimension in self.homologyDimensions ]
            XTransformed = Parallel(n_jobs = self.n_jobs) ( delayed(self._pad_diagram)(XTransformed[i], maxLengthList)
                                                            for i in range(len(XTransformed)) )
            XTransformed = self._stack_padded_diagrams(XTransformed)

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed
