"""The module :mod:`gtda.plotting` implements function to plot giotto-tda
transformers' output."""

from .point_clouds import plot_point_cloud
from .homology import plot_diagram
from .diagrams import plot_betti_curves, plot_betti_surfaces, plot_landscapes,\
    plot_persistence_image, plot_silhouettes, plot_heat_kernel
from .image import ImagePlotterMixin

__all__ = [
    'plot_point_cloud',
    'plot_diagram',
    'plot_betti_curves',
    'plot_betti_surfaces',
    'plot_landscapes',
    'plot_persistence_image',
    'plot_silhouettes',
    'plot_heat_kernel',
    'ImagePlotterMixin',
]
