"""The module :mod:`gtda.plotting` implements function to plot giotto-tda
transformers' output."""

from .point_clouds import plot_point_cloud, PointCloudPlotter
from .homology import plot_diagram, HomologyPlotter
from .diagrams import plot_betti_curves, plot_betti_surfaces, plot_landscapes,\
    plot_persistence_image, plot_silhouettes, plot_heat_kernel
from .image import ImagePlotter

__all__ = [
    'plot_point_cloud',
    'plot_diagram',
    'plot_betti_curves',
    'plot_betti_surfaces',
    'plot_landscapes',
    'plot_persistence_image',
    'plot_silhouettes',
    'plot_heat_kernel',
    'ImagePlotter',
    'PointCloudPlotter',
    'HomologyPlotter'
]
