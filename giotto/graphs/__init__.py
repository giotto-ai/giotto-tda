"""The module :mod:`giotto.graphs` implements transformers to create graphs or
extract metric spaces from graphs.
"""

from giotto.graphs.geodesic_distance import GraphGeodesicDistance
from giotto.graphs.kneighbors import KNeighborsGraph
from giotto.graphs.transition import TransitionGraph

import warnings

warnings.warn(
    "Starting at v0.1.4, this package was renamed as 'giotto-tda'. The "
    "giotto-learn PyPI package will no longer be developed or maintained, and "
    "will remain at the state of v0.1.3. Please visit "
    "https://github.com/giotto-ai/giotto-tda to find installation information "
    "for giotto-tda.")


__all__ = [
    'TransitionGraph',
    'KNeighborsGraph',
    'GraphGeodesicDistance'
]
