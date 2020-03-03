"""Utilities for input validation."""
# License: GNU AGPLv3

import numpy as np


def check_diagram(X, copy=False):
    """Input validation on a persistence diagram.

    """
    if len(X.shape) != 3:
        raise ValueError(f"X should be a 3d np.array: X.shape = {X.shape}.")
    if X.shape[2] != 3:
        raise ValueError(
            f"X should be a 3d np.array with a 3rd dimension of 3 components: "
            f"X.shape[2] = {X.shape[2]}.")

    homology_dimensions = sorted(list(set(X[0, :, 2])))
    for dim in homology_dimensions:
        if dim == np.inf:
            if len(homology_dimensions) != 1:
                raise ValueError(
                    f"np.inf is a valid homology dimension for a stacked "
                    f"diagram but it should be the only one: "
                    f"homology_dimensions = {homology_dimensions}.")
        else:
            if dim != int(dim):
                raise ValueError(
                    f"All homology dimensions should be integer valued: "
                    f"{dim} can't be cast to an int of the same value.")
            if dim != np.abs(dim):
                raise ValueError(
                    f"All homology dimensions should be integer valued: "
                    f"{dim} can't be cast to an int of the same value.")

    n_points_above_diag = np.sum(X[:, :, 1] >= X[:, :, 0])
    n_points_global = X.shape[0] * X.shape[1]
    if n_points_above_diag != n_points_global:
        raise ValueError(
            f"All points of all persistence diagrams should be above the "
            f"diagonal, X[:,:,1] >= X[:,:,0]. "
            f"{n_points_global - n_points_above_diag} points are under the "
            f"diagonal.")
    if copy:
        return np.copy(X)
    else:
        return X


def check_graph(X):
    return X


def _validate_params_single(parameter, reference, name):
    if reference is None:
        return

    ref_type = reference.get('type', None)

    # Check that parameter has the correct type
    if (ref_type is not None) and (not isinstance(parameter, ref_type)):
        raise TypeError(
            f"Parameter `{name}` is of type {type(parameter)} while "
            f"it should be of type {ref_type}.")

    # If the reference type parameter is not list or tuple, the checks are
    # performed on the parameter object directly. Note: ref_type can be
    elif ref_type not in [list, tuple, np.ndarray, dict]:
        ref_in = reference.get('in', None)
        ref_other = reference.get('other', None)
        if parameter is not None:
            if (ref_in is not None) and (parameter not in ref_in):
                raise ValueError(
                    f"Parameter `{name}` is {parameter}, which is not in "
                    f"{ref_in}.")
        # Perform any other checks via the callable ref_others
        if ref_other is not None:
            return ref_other(parameter)

    # Explicitly return a boolean flag to indicate that checks must be
    # performed recursively at deeper layers
    else:
        return ref_type


def _validate_params(parameters, references, rec_name=None):
    for name, parameter in parameters.items():
        if name == 'n_jobs':
            continue
        if name not in references.keys():
            name_extras = "" if rec_name is None else f" in `{rec_name}`"
            raise KeyError(
                f"`{name}`{name_extras} is not an available parameter. "
                f"Available parameters are in {list(references.keys())}.")

        reference = references[name]
        ref_type = _validate_params_single(parameter, reference, name)
        if ref_type:
            ref_of = reference.get('of', None)
            if ref_type == dict:
                _validate_params(parameter, ref_of, rec_name=name)
            else:  # List or tuple type
                for i, parameter_elem in enumerate(parameter):
                    _validate_params_single(
                        parameter_elem, ref_of, f"{name}[{i}]")


def validate_params(parameters, references, exclude=None):
    """Function to automate the validation of hyperparameters.

    Parameters
    ----------
    parameters : dict, required
        Dictionary in which the keys are hyperparameter names and the
        corresponding values are hyperparameter values.

    references : dict, required
        Dictionary in which the keys are hyperparameter names and the
        corresponding values are lists. The first element of that list is a
        type. If that type is not one of ``list``, ``tuple`` or
        ``numpy.ndarray``, the second element can be one of:

        - ``None``, when only the type should be checked.
        - A tuple of two numbers, when the type is numerical. In this case,
          the first (resp. second) entry in the tuple defines a lower
          (resp. upper) bound constraining the value of the corresponding
          hyperparameter.
        - A list containing all possible allowed values for the
          corresponding hyperparameter.

        If that type is one of ``list``,``tuple`` or ``numpy.ndarray``,
        the second element is a list that provides information to validate
        each element of that iterable. The first element of that list is the
        type of the elements of the iterable and the second element of that
        list can be one of:

        - ``None``, when only the type of the iterable elements should be
          checked.
        - A tuple of two numbers, when the type is numerical. In this case,
          the first (resp. second) entry in the tuple defines a lower
          (resp. upper) bound constraining the value of the corresponding
          iterable element.
        - A list containing all possible allowed values for the
          corresponding iterable element.

    """
    exclude_ = [] if exclude is None else exclude
    parameters_ = {key: value for key, value in parameters.items()
                   if key not in exclude_}
    return _validate_params(parameters_, references)
