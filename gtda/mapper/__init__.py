"""The module :mod:`gtda.mapper` implements the Mapper algorithm for
topological clustering and visualisation."""

from .cluster import FirstHistogramGap, FirstSimpleGap, ParallelClustering
from .cover import CubicalCover, OneDimensionalCover
from .filter import Eccentricity, Entropy, Projection
from .nerve import Nerve
from .pipeline import make_mapper_pipeline
from .utils.decorators import method_to_transform
from .utils.pipeline import transformer_from_callable_on_rows
from .visualization import plot_static_mapper_graph, \
    plot_interactive_mapper_graph, MapperInteractivePlotter

__all__ = [
    'Projection',
    'Eccentricity',
    'Entropy',
    'OneDimensionalCover',
    'CubicalCover',
    'FirstSimpleGap',
    'FirstHistogramGap',
    'ParallelClustering',
    'Nerve',
    'make_mapper_pipeline',
    'plot_static_mapper_graph',
    'plot_interactive_mapper_graph',
    'MapperInteractivePlotter',
    'method_to_transform',
    'transformer_from_callable_on_rows'
    ]
