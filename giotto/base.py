# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: TBD

from .neural_network.keras import KerasClassifierWrapper, KerasRegressorWrapper


def clone(estimator, safe=True):
    if (isinstance(estimator, KerasClassifierWrapper) or
            isinstance(estimator, KerasRegressorWrapper)):
        return estimator.clone()
    else:
        from sklearn.base import clone as sklearn_clone
        return sklearn_clone(estimator, safe)
