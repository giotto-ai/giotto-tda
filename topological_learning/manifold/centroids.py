import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.ensemble

import numpy as np
from .DiagramDistance import DiagramDistance


class CentroidsDistance(BaseEstimator, TransformerMixin):
    def __init__(self, distance_kwargs = {}):
        self.distance_kwargs = distance_kwargs

    def get_params(self, deep=True):
        return {'distance_kwargs': self.distance_kwargs}

    def fit(self, XList, y):
        self.isFitted = True

        self.distance = DiagramDistance(**self.distance_kwargs)
        XDistance = self.distance.fit(XList).transform(XList)[0]

        yInt = np.rint(XList[1])
        labels = np.unique(yInt, axis=0)

        localIndex = []
        globalIndex = []

        currentLabelNumber = 0
        for label in labels:
            yLabel = np.argwhere(np.all(yInt == label, axis =1)).flatten()
            XLabel = XDistance[np.ix_(yLabel, yLabel)]
            XDegree =  np.sum(XLabel, axis = 1)
            XDegree = np.where(XDegree > 0, XDegree, np.inf)
            localIndex = np.argmin(XDegree)

            count = -1
            for i in range(len(yInt)):
                if (yInt[i] == label).all() :
                    count += 1
                if count == localIndex:
                    globalIndex.append(i)
                    break
            currentLabelNumber += 1

        XCentroids = [ XList[0][index] for index in globalIndex ]
        self.numberCentroids = len(XCentroids)
        self.distance.fit([ XCentroids ])

        # Return the classifier
        return self

    def transform(self, XList, y = None):
        # Check is fit had been called
        check_is_fitted(self, ['isFitted'])

        # Input validation
        #  X = check_array(X)

        XListTransformed = []

        XTransformed = self.distance.transform(XList)[0][:self.numberCentroids,:] #.T ??????

        XListTransformed.append(XTransformed)

        if len(XList) == 2:
            XListTransformed.append(XList[1])

        return XListTransformed
