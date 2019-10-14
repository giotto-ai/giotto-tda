"""
The :mod:`giotto.utils` module includes various utilities.
"""

from .validation import check_diagram, check_graph
from .validation import validate_metric_params, validate_params


__all__ = [
    'check_diagram',
    'check_graph',
    'validate_metric_params',
    'validate_params'
]
