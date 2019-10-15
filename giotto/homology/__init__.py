"""The module :mod:`giotto.homology` implements transformers
to generate features from persistent homology.
"""

from .consistent import ConsistentRescaling
from .point_clouds import VietorisRipsPersistence

__all__ = [
    'VietorisRipsPersistence',
    'ConsistentRescaling',
]
