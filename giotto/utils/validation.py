"""Utilities for input validation"""
import numpy as np
import numbers

def check_diagram(X):
    """Input validation on a diagram
    """
    _n_outer_window = list(X.values())[0].shape[0]
    for z in X.keys():
        if isinstance(z, numbers.Number):
            if (z < 0):
                raise ValueError("X key has the wrong value: {}. "
                                "X keys must be non-negative "
                                "integers.".format(z))
        else:
            raise TypeError("X key has the wrong type: %s. "
                            "X keys must be non-negative "
                            "integers." % type(z))
        if (_n_outer_window != X[z].shape[0]):
            raise ValueError("Diagram first dimension has the wrong value: {}."
                             " Diagram first dimension must be equal "
                             "for all subarrays.".format(X[z].shape[0]))
        if (len(X[z].shape) != 3):
            raise ValueError("Diagram structure dimension error: {}. "
                             "Diagram structure dimension must be equal "
                             "to 3.".format(len(X[z].shape)))
        if (X[z].shape[2] != 2):
            raise ValueError("Wrong dimension for persistent "
                             "diagram coordinates: {}. "
                             "Diagram coordinates dimension must be equal "
                             "to 2.".format(X[z].shape[2]))
        _diff_coord = (X[z].shape[0]*X[z].shape[1] - 
                      np.sum(X[z][:, :, 1] >= X[z][:, :, 0]))
        if (_diff_coord > 0):
            raise ValueError("Coordinates have the wrong value: {} of "
                             "them are wrong. They must be "
                             "integers and the 2nd must be greater than "
                             "or equal to the 1st one.".format(_diff_coord))
