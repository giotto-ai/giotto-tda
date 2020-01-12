"""The module :mod:`giotto.homology` implements transformers
to modify metric spaces or generate persistence diagrams.
"""

from .consistent import ConsistentRescaling
from .point_clouds import VietorisRipsPersistence

__all__ = [
    'VietorisRipsPersistence',
    'ConsistentRescaling',
]
