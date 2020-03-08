"""Utilities for input validation."""
# License: GNU AGPLv3

import numpy as np
import types

from sklearn.utils.validation import check_array

def check_diagram(X, copy=False):
    """Input validation on a persistence diagram.

    """
    if X.ndim != 3:
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

    # If the reference type parameter is not list, tuple, np.ndarray or dict,
    # the checks are performed on the parameter object directly.
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

    # Explicitly return the type of reference if one of list, tuple, np.ndarray
    # or dict.
    else:
        return ref_type


def _validate_params(parameters, references, rec_name=None):
    for name, parameter in parameters.items():
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
            else:  # List, tuple or ndarray type
                for i, parameter_elem in enumerate(parameter):
                    _validate_params_single(
                        parameter_elem, ref_of, f"{name}[{i}]")


def validate_params(parameters, references, exclude=None):
    """Function to automate the validation of (hyper)parameters.

    Parameters
    ----------
    parameters : dict, required
        Dictionary in which the keys parameter names (as strings) and the
        corresponding values are parameter values. Unless `exclude` (see
        below) contains some of the keys in this dictionary, all parameters
        are checked against `references`.

    references : dict, required
        Dictionary in which the keys are parameter names (as strings). Let
        ``name`` and ``parameter`` denote a key-value pair in `parameters`.
        Since ``name`` should also be a key in `references`, let ``reference``
        be the corresponding value there. Then, ``reference`` must be a
        dictionary containing any of the following keys:

        - ``'type'``, mapping to a class or tuple of classes. ``parameter``
          is checked to be an instance of this class or tuple of classes.

        - ``'in'``, mapping to a dictionary, when the value of ``'type'`` is
          not one of ``list``, ``tuple``, ``numpy.ndarray`` or ``dict``.
          Letting ``ref_in`` denote that dictionary, the following check is
          performed: ``parameter in ref_in``.

        - ``'of'``, mapping to a dictionary, when the value of ``'type'``
          is one of ``list``, ``tuple``, ``numpy.ndarray`` or ``dict``.
          Let ``ref_of`` denote that dictionary. Then:

          a) If ``reference['type'] == dict`` – meaning that ``parameter``
             should be a dictionary – ``ref_of`` should have a similar
             structure as `references`, and :func:`validate_params` is called
             recursively on ``(parameter, ref_of)``.
          b) Otherwise, ``ref_of`` should have a similar structure as
             ``reference`` and each entry in ``parameter`` is checked to
             satisfy the constraints in ``ref_of``.

        - ``'other'``, which should map to a callable defining custom checks on
          ``parameter``.

    exclude : list of str, or None, optional, default: ``None``
        List of parameter names which are among the keys in `parameters` but
        should be excluded from validation. ``None`` is equivalent to
        passing the empty list.

    """
    exclude_ = [] if exclude is None else exclude
    parameters_ = {key: value for key, value in parameters.items()
                   if key not in exclude_}
    return _validate_params(parameters_, references)


def check_list_of_images(X, **kwargs):
    """Check a list of arrays representing images, by integrating
    through the input one by one. To pass a test when `kwargs` is ``None``,
    all images ``x``, ``y`` in `X` must satisfy:
        - ``x.ndim >= 2``,
        - ``all(np.isfinite(x))``,
        - ``x.shape == y.shape``.

    Parameters
    ----------
    X : list of ndarray
        Each entry of `X` corresponds to an image.

    kwargs : dict or None, optional, default: ``None``
        Parameters accepted by
        :func:`~gtda.utils.validation.check_list_of_arrays`.

    Returns
    -------
    X : list of ndarray
        as modified by :func:`~sklearn.utils.validation.check_array`

    """
    if hasattr(X, 'shape'):
        if X.ndim < 3:
            raise ValueError(f"An image in the collection X should be at "
                             f"least of dimension 2, while it has dimension "
                             f"{X.ndim - 1}.")
        return check_array(X, **kwargs)
    else:
        kwargs_default = {'force_all_finite': True,
                          'ensure_2d': False, 'allow_nd': True,
                          'check_shapes': [('embedding_dimension',
                                            lambda x: x.shape,
                                            'The images should have exactly'
                                            'the same shape')]}
        kwargs_default.update(kwargs)
        return check_list_of_arrays(X, **kwargs_default)


def check_list_of_point_clouds(X, **kwargs):
    """Check a list of arrays representing point clouds, by integrating
    through the input one by one. To pass a test when `kwargs` is ``None``,
    all point clouds ``x``, ``y`` in X must satisfy:
        - ``x.ndim == 2``,
        - ``len(y.shape[1:]) == len(y.shape[1:])``.

    Parameters
    ----------
    X : list of ndarray, such that ``X[i].ndim==2`` (n_points, n_dimensions),
        or an array `X.dim==3`

    kwargs : dict or None, optional, default: ``None``
        Parameters accepted by
        :func:`~`gtda.utils.validation.check_list_of_arrays``.

    Returns
    -------
    X : list of input arrays
        as modified by :func:`~sklearn.utils.validation.check_array`

    """
    if hasattr(X, 'shape'):
        return check_array(X, **kwargs)
    else:
        kwargs_default = {'ensure_2d': False, 'force_all_finite': False,
                          'check_shapes': [('embedding_dimension',
                                            lambda x: x.shape[1:],
                                            'Not all point clouds have the '
                                            'same embedding dimension.')]}
        kwargs_default.update(kwargs)
        return check_list_of_arrays(X, **kwargs_default)


def check_dimensions(X, get_property):
    """Check the dimensions of X are consistent, where the check is defined
    by get_property 'sample-wise'.
    Parameters
    ----------
    X: list of ndarray,
        Usually represents point clouds or images- see
        :func:`~`gtda.utils.validation.check_list_of_arrays``.

    get_property: function: ndarray -> _,
        Defines a property to be conserved, across all arrays (samples)
        in X.

    """
    from functools import reduce
    from operator import and_
    reference = get_property(X[0])
    return reduce(and_, map(lambda x: get_property(x) == reference, X[1:]),
                  True)


def check_list_of_arrays(X, check_shapes=list(), **kwargs):
    """Input validation on a list of lists, arrays, sparse matrices, or similar.

    The constraints are to be specified in `kwargs`. On top of
    parameters from :func:`~sklearn.utils.validation.check_array`,
    the optional parameters are listed below.

    Parameters
    ----------
    X : list
        Input list of objects to check / convert.

    check_shapes: list of tuples t, where t = (str, function to pass to
        check_dimensions, error message if test fails).
        The checks are applied in the order they are provided, only until
        the first failure.

    kwargs : dict or None, optional, default: ``None``
        Parameters accepted by :func:`~sklearn.utils.validation.check_array`.

    Returns
    -------
    X : list
        Output list of objects, each checked / converted by
        :func:`~sklearn.utils.validation.check_array`

    """

    # if restrictions on the dimensions of the input are imposed
    for (test_name, get_property, err_message) in check_shapes:
        if check_dimensions(X, get_property):
            continue
        else:
            raise ValueError(err_message)

    is_check_failed = False
    messages = []
    for i, x in enumerate(X):
        try:
            # TODO: verifythe behavior depending on copy.
            X[i] = check_array(x.reshape(1, *x.shape),
                               **kwargs).reshape(*x.shape)
            messages = ['']
        except ValueError as e:
            is_check_failed = True
            messages.append(str(e))
    if is_check_failed:
        raise ValueError("The following errors were raised" +
                         "by the inputs: \n" + "\n".join(messages))
    else:
        return X
