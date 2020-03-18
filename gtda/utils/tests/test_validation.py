"""Tests for validation functions."""
# License: GNU AGPLv3

import numpy as np
import pytest

from gtda.utils.validation import check_diagrams, validate_params


# Testing for validate_params
def test_validate_params():
    """These tests should fail because either the type of parameters[
    parameter_name] is incorrect, or because parameter not in references[
    parameter_name]['in']."""
    references = {'par1': {'type': int, 'in': [0, 1]}}
    parameters = {'par1': 0.5}

    with pytest.raises(TypeError):
        validate_params(parameters, references)

    parameters = {'par1': 2}
    with pytest.raises(ValueError):
        validate_params(parameters, references)

    parameters = {'par0': 1}
    with pytest.raises(KeyError):
        validate_params(parameters, references)


# Testing for validate_params when one of the parameters is of list type
def test_validate_params_list():
    """Test the behaviour of validate_params on parameters which are of list
    type. Each entry in the list should satisfy the constraints described by
    references[parameter_name]['of']."""
    references = {
        'par1': {'type': list, 'of': {'type': float, 'in': [1., 2.]}}
    }
    parameters = {'par1': [1.]}

    validate_params(parameters, references)


# Testing check_diagrams
# Test for the wrong array key value
def test_inputs_keys_V():
    X = np.array([[[1, 1, 0], [2, 2, -1]]])
    with pytest.raises(ValueError):
        check_diagrams(X)


# Test for the wrong structure dimension
def test_inputs_arrayStruc_V():
    X = np.array([[[[1, 1, 0], [2, 2, 1]]]])

    with pytest.raises(ValueError):
        check_diagrams(X)
