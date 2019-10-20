"""Tests for validation functions"""
# License: Apache 2.0

import numpy as np
import pytest

from ..validation import check_diagram
from ..validation import validate_metric_params, validate_params


# Testing for validate_params
def test_validate_params():
    references = {'par1': [int, [0, 1]]}
    parameters = {'par1': 0.5}

    with pytest.raises(TypeError):
        validate_params(parameters, references)

    parameters = {'par1': 2}
    with pytest.raises(ValueError):
        validate_params(parameters, references)

    parameters = {'par0': 1}
    with pytest.raises(KeyError):
        validate_params(parameters, references)


# Testing check_diagram
# Test for the wrong array key value
def test_inputs_keys_V():
    X = np.array([[[1, 1, 0], [2, 2, -1]]])
    with pytest.raises(ValueError):
        check_diagram(X)


# Test for the wrong structure dimension
def test_inputs_arrayStruc_V():
    X = np.array([[[[1, 1, 0], [2, 2, 1]]]])

    with pytest.raises(ValueError):
        check_diagram(X)


# Testing validate_metric_params
# Test for the wrong metric value
def test_metric_V():
    with pytest.raises(ValueError, match="No metric called"):
        validate_metric_params('bottleeck', metric_params={
            'n_values': 200, 'delta': 0.01})


# Test for the wrong n_values type
def test_n_values_T():
    with pytest.raises(TypeError, match=" in params_metric is of type "):
        validate_metric_params('landscape',
                               metric_params={'n_values': 'a',
                                              'delta': 0.01})


# Test for the wrong n_values value
def test_n_values_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('landscape',
                               metric_params={'n_values': -2,
                                              'delta': 0.01})


# Test for the wrong delta value
def test_delta_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('bottleneck',
                               metric_params={'n_values': 200,
                                              'delta': -1})


# Test for the wrong delta value
def test_delta_T():
    with pytest.raises(TypeError, match=" in params_metric is of type"):
        validate_metric_params('bottleneck',
                               metric_params={'n_values': 200,
                                              'delta': 'a'})


# Test for the wrong order value
def test_order_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('heat',
                               metric_params={'n_values': 200,
                                              'order': -1})


# Test for the wrong order type
def test_order_T():
    with pytest.raises(TypeError, match=" in params_metric is of type"):
        validate_metric_params('heat',
                               metric_params={'n_values': 200,
                                              'order': 'a'})


# Test for the wrong sigma value
def test_sigma_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('heat',
                               metric_params={'n_values': 200,
                                              'sigma': -1})


# Test for the wrong sigma type
def test_sigma_T():
    with pytest.raises(TypeError, match=" in params_metric is of type"):
        validate_metric_params('heat', metric_params={'n_values': 200,
                                                      'sigma': 'a'})


# Undefined metric_params
def test_validate():
    with pytest.raises(ValueError):
        validate_metric_params('heat', metric_params={'blah': 200})


# Testing check_graph
# Test for the dimension
def test_check_graph_structure_V():
    X = np.array([[0, 1, 3, 0, 0], [1, 0, 5, 0, 0], [3, 5, 0, 4, 0],
                  [0, 0, 4, 0, 0], [0, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="Graph structure dimension must "
                       "be equal to 3."):
        check_graph(X)

# Test for the diagonal elements
def test_check_graph_diagonal_V():
    X = np.array([np.array([[-1, 1, 3, 0, 0], [1, 0, 5, 0, 0],
                            [3, 5, 0, 4, 0], [0, 0, 4, 0, 0],
                            [0, 0, 0, 0, 0]])])
    with pytest.raises(ValueError, match="At least one matrix diagonal "
                       "element is not 0"):
        check_graph(X)

# Test to check if matrices are square matrices
def test_check_graph_square_T():
    X = np.array([np.array([[0, 1, 3, 0, 0], [1, 0, 5, 0, 0], [0, 0, 4, 0, 0],
                            [0, 0, 0, 0, 0]])])
    with pytest.raises(TypeError, match="Matrix is not square"):
        check_graph(X)

# Test to check if matrices are symmetric
def test_check_graph_symmetric_V():
    X = np.array([np.array([[0, 1, 3, 0, 0], [1, 0, 5, 0, 0],
                            [3, 5, 0, 4, 0], [0, 0, 4, 0, 0],
                            [0, 8, 0, 0, 0]])])
    with pytest.raises(ValueError, match="Matrix is not symmetric"):
        check_graph(X)

# Test to check if all the elements are greater than or equal to 0
def test_check_graph_element_V():
    X = np.array([np.array([[0, 1, 3, 0, -1], [1, 0, 5, 0, 0],
                            [3, 5, 0, 4, 0], [0, 0, 4, 0, 0],
                            [-1, 0, 0, 0, 0]])])
    with pytest.raises(ValueError, match="At least one matrix element "
                       "is smaller than 0"):
        check_graph(X)
