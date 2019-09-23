"""Utilities for input validation"""
import numbers

import numpy as np

available_metrics = {'bottleneck': [('delta', numbers.Number, (0., 1.))],
                     'wasserstein': [('order', int, (1, np.inf)),
                                     ('delta', numbers.Number, (0., 1.))],
                     'betti': [('order', int, (1, np.inf)),
                               ('n_samples', int, (1, np.inf))],
                     'landscape': [('order', int, (1, np.inf)),
                                   ('n_samples', int, (1, np.inf)),
                                   ('n_layers', int, (1, np.inf))],
                     'heat': [('order', int, (1, np.inf)),
                              ('n_samples', int, (1, np.inf)),
                              ('sigma', numbers.Number, (0., np.inf))]}

available_metric_params = list(set(
    [param for param_list in available_metrics.values()
     for (param, param_type, param_range) in param_list]))


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
        _diff_coord = (X[z].shape[0] * X[z].shape[1] -
                       np.sum(X[z][:, :, 1] >= X[z][:, :, 0]))
        if (_diff_coord > 0):
            raise ValueError("Coordinates have the wrong value: {} of "
                             "them are wrong. They must be "
                             "integers and the 2nd must be greater than "
                             "or equal to the 1st one.".format(_diff_coord))
    return X


def validate_metric_params(metric, metric_params):
    if (metric not in available_metrics.keys()):
        raise ValueError("No metric called {}."
                         " Available metrics are {}."
                         "".format(metric,
                                   list(available_metrics.keys())))

    for (param, param_type, param_values) in available_metrics[metric]:
        if param in metric_params.keys():
            input_param = metric_params[param]
            if not isinstance(input_param, param_type):
                raise TypeError("{} in params_metric is of type {}"
                                " but must be an {}."
                                "".format(param, type(input_param),
                                          param_type))

            if input_param < param_values[0] or input_param > param_values[1]:
                raise ValueError("{} in param_metric should be between {} "
                                 "and {} but has been set to {}."
                                 "".format(param, param_values[0],
                                           param_values[1], input_param))

    for param in metric_params.keys():
        if param not in available_metric_params:
            raise ValueError("{} in param_metric is not an available"
                             " parameter. Available metric_params."
                             " are {}".format(param,
                                              available_metric_params))
