import numpy as np
import pytest

from ..validation import check_diagram
from ..validation import validate_metric_params

'''Tests for validation functions'''


# Testing check_diagram
# Test for the wrong array key value
def test_inputs_keys_V():
    X = {
        0: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        -1: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        # Wrong array key
        3: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]])
    }
    with pytest.raises(ValueError, match="X keys must be non-negative "
                                         "integers."):
        check_diagram(X)


# Test for the wrong array key type
def test_inputs_keys_T():
    X = {
        0: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        'a': np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        # Wrong array key
        3: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]])
    }
    with pytest.raises(TypeError, match="X keys must be non-negative "
                                        "integers."):
        check_diagram(X)


# Test for the wrong structure dimension
def test_inputs_arrayStruc_V():
    X = {
        0: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        1: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        3: np.array([[[1, 1], [2, 2]], [[4, 4], [5, 5], [6, 6]]])
        # Wrong array structure dimension
    }
    with pytest.raises(ValueError,
                       match="Diagram structure dimension must be equal "
                             "to 3."):
        check_diagram(X)


# Test for the wrong 1st array dimension
def test_inputs_arraydim1_V():
    X = {
        0: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        1: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        3: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]],
                     [[4, 4], [5, 5], [6, 6]]])  # Wrong array 1st dimension
    }
    with pytest.raises(ValueError, match="Diagram first dimension must "
                                         "be equal for all subarrays."):
        check_diagram(X)


# Test for the wrong 3rd array dimension
def test_inputs_arraydim3_V():
    X = {
        0: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        1: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        3: np.array([[[1], [2], [3]], [[4], [5], [6]]])
        # Wrong array 3rd dimension
    }
    with pytest.raises(ValueError, match=" Diagram coordinates dimension "
                                         "must be equal to 2."):
        check_diagram(X)


# Test for the wrong value of a 3rd dimension array's elements
def test_inputs_dim3_coord_V():
    X = {
        0: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        1: np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]),
        3: np.array([[[1, 1], [2, 2], ['b', 3]], [[4, 4], [5, 5], [6, 6]]])
        # Wrong array element value
    }
    with pytest.raises(ValueError, match="They must be "
                                         "integers and the 2nd must be greater than "
                                         "or equal to the 1st one."):
        check_diagram(X)


# Testing validate_metric_params
# Test for the wrong metric value
def test_metric_V():
    with pytest.raises(ValueError, match="No metric called"):
        validate_metric_params('bottleeck', metric_params={'n_samples': 200,
                                                           'delta': 0.0})


# Test for the wrong n_sample type
def test_n_samples_T():
    with pytest.raises(TypeError, match=" in params_metric is of type "):
        validate_metric_params('landscape',
                               metric_params={'n_samples': 'a',
                                              'delta': 0.0})


# Test for the wrong n_sample value
def test_n_samples_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('landscape',
                               metric_params={'n_samples': -2,
                                              'delta': 0.0})


# Test for the wrong delta value
def test_delta_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('bottleneck',
                               metric_params={'n_samples': 200,
                                              'delta': -1}, )


# Test for the wrong delta value
def test_delta_T():
    with pytest.raises(TypeError, match=" in params_metric is of type"):
        validate_metric_params('bottleneck',
                               metric_params={'n_samples': 200,
                                              'delta': 'a'})


# Test for the wrong order value
def test_order_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('heat',
                               metric_params={'n_samples': 200,
                                              'order': -1})


# Test for the wrong order type
def test_order_T():
    with pytest.raises(TypeError, match=" in params_metric is of type"):
        validate_metric_params('heat',
                               metric_params={'n_samples': 200,
                                              'order': 'a'})


# Test for the wrong sigma value
def test_sigma_V():
    with pytest.raises(ValueError, match=" in param_metric should be between"):
        validate_metric_params('heat',
                               metric_params={'n_samples': 200,
                                              'sigma': -1})


# Test for the wrong sigma type
def test_sigma_T():
    with pytest.raises(TypeError, match=" in params_metric is of type"):
        validate_metric_params('heat', metric_params={'n_samples': 200,
                                                      'sigma': 'a'})


# Undefined metric_params
def test_validate():
    with pytest.raises(ValueError):
        validate_metric_params('heat', metric_params={'blah': 200})
