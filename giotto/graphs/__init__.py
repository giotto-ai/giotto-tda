"""The :mod:`giotto.graphs` implements preprocessing techniques
   useful to apply Topological Data Analysis techniques to graphs.
"""

from .preprocessing import TransitionGraph, kNNGraph, GraphGeodesicDistance


__all__ = [
    'TransitionGraph',
    'kNNGraph',
    'GraphGeodesicDistance'
]
