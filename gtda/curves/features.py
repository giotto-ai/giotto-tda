"""Feature extraction from curves."""
# License: GNU AGPLv3

from inspect import signature
from types import FunctionType

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from ._functions import _AVAILABLE_FUNCTIONS, _implemented_function_recipes, \
    _parallel_featurization
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class StandardFeatures(BaseEstimator, TransformerMixin):
    """Standard features from multi-channel curves.

    Applies functions to extract features from each channel in each
    multi-channel curve in a collection.

    Parameters
    ----------
    function : string or callable, optional, default: ``max``
        Function to transform a single-channel curve into scalar features per
        channel. Implemented functions are [``"argmin"``, `"argmax"``,
        ``"min"``, ``"max"``, ``"mean"``, ``"std"``, ``"median"``,
        ``"average"``].

    function_params : dict, optional, default: ``None``
        Additional keyword arguments for `function`. Passing ``None`` is
        equivalent to passing no arguments. Additionally:

        - If ``function == "average"``, the only argument is `weights`
          (np.ndarray or None, default: ``None``).
        - Otherwise, there are no arguments.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_channels_ : int

    function_ : callable
        Function used to transform single-channel curves into scalar
        features. Set in :meth:`fit`.

    effective_function_params_ : dict
        Dictionary containing all information present in `function_params` as
        well as on any relevant quantities computed in :meth:`fit`.

    """
    _hyperparameters = {
        "function": {"type": (str, FunctionType, list, tuple),
                     "in": tuple(_AVAILABLE_FUNCTIONS.keys()),
                     "of": {"type": (str, FunctionType, type(None)),
                            "in": tuple(_AVAILABLE_FUNCTIONS.keys())}},
        "function_params": {"type": (dict, type(None))},
        }

    def __init__(self, function="max", function_params=None, n_jobs=None):
        self.function = function
        self.function_params = function_params
        self.n_jobs = n_jobs

    def _validate_params(self):
        try:
            validate_params(
                self.get_params(), self._hyperparameters, exclude=["n_jobs"])
        # Another go if we fail because function is or contains instances of
        # FunctionType and the "in" key checks fail
        except ValueError:
            _hyperparameters = self._hyperparameters.copy()
            _hyperparameters["function"].pop("in")
            _hyperparameters["function"]["of"].pop("in")
            validate_params(
                self.get_params(), _hyperparameters, exclude=["n_jobs"])

    def fit(self, X, y=None):
        """Compute :attr:`function_` and :attr:`effective_function_params_`.
        Then, return the estimator.

        This function is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_bins)
            Input data. Collection of multi-channel curves.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, allow_nd=True)
        if X.ndim != 3:
            raise ValueError("Input must be 3-dimensional with shape "
                             "(n_samples, n_channels, n_bins).")
        self._validate_params()

        self.n_channels_ = X.shape[-2]

        if isinstance(self.function, str):
            self._function = _implemented_function_recipes[self.function]
            self.function_ = tuple([self._function] * self.n_channels_)

            if self.function_params is None:
                self.effective_function_params_ = {}
            else:
                self.effective_function_params_ = self.function_params.copy()
            validate_params(self.effective_function_params_,
                            _AVAILABLE_FUNCTIONS[self.function])
        else:
            if isinstance(self.function, FunctionType):
                self.function_ = tuple([self.function] * self.n_channels_)
                if "axis" in signature(self.function).parameters:
                    self._function = self.function
                else:
                    self._function = self.function_
            else:
                n_functions = len(self.function)
                if len(self.function) != self.n_channels_:
                    raise ValueError(
                        f"`function` has length {n_functions} while curves in "
                        f"`X` have {self.n_channels_} channels."
                        )
                self.function_ = tuple(self.function)
                self._function = self.function_
            self.effective_function_params_ = {}

        return self

    def transform(self, X, y=None):
        """Compute features of multi-channel curves.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_bins)
            Input collection of multi-channel curves.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_channels * n_features)
            Output collection of curves features. ``n_features`` denotes the
            number of features output byt :attr:`function_` for each channel of
            the multi-channel curve.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = _parallel_featurization(Xt, self._function,
                                     self.effective_function_params_,
                                     self.n_jobs)

        return Xt
