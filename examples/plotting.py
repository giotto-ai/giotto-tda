"""Plot functions """

import numpy as np
import plotly.graph_objs as gobj
from giotto.diagrams._utils import _subdiagrams


def plot_point_cloud(point_cloud, dimension=None):
    """Plot the first 2 or 3 coordinates of the point cloud.

     This function will not work on 1-dimensional arrays.

    Parameters
    ----------
    point_cloud : ndarray, shape (n_samples, n_dimensions)
        Data points to be represented in a 2D or 3D scatter plot. Only the
        first 2 or 3 dimensions will be considered for plotting.

    dimension : int or None, default : ``None``
        This parameter sets the dimension of the resulting plot. If ``None``,
        the dimension will be chosen between 2 and 3 depending on
        ``n_dimensions`` see Input).

    """
    if dimension is None:
        dimension = np.min((3, point_cloud.shape[1]))

    # Check consistency between point_cloud and dimension
    if point_cloud.shape[1] < dimension:
        raise ValueError("Not enough dimensions available in the input point"
                         "cloud.")

    if dimension == 2:
        layout = {
            "title": "Point Cloud",
            "width": 800,
            "height": 800,
            "xaxis1": {
                "title": "First coordinate",
                "side": "bottom",
                "type": "linear",
                "ticks": "outside",
                "anchor": "x1",
                "showline": True,
                "zeroline": True,
                "showexponent": "all",
                "exponentformat": "e"
            },
            "yaxis1": {
                "title": "Second coordinate",
                "side": "left",
                "type": "linear",
                "ticks": "outside",
                "anchor": "y1",
                "showline": True,
                "zeroline": True,
                "showexponent": "all",
                "exponentformat": "e"
            },
            "plot_bgcolor": "white"
        }
    
        fig = gobj.Figure(layout=layout)
        fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                         mirror=False)
        fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                         mirror=False)

        fig.add_trace(gobj.Scatter(x=point_cloud[:, 0],
                                   y=point_cloud[:, 1],
                                   mode='markers',
                                   marker=dict(size=4,
                                               color=list(range(
                                                   point_cloud.shape[0])),
                                               colorscale='Viridis',
                                               opacity=0.8)))
        fig.show()
    elif dimension == 3:
        
        scene = {
            "xaxis": {
                "title": "First coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
            },
            "yaxis": {
                "title": "Second coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
            },
            "zaxis": {
                "title": "Third coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
            }
        }

        fig = gobj.Figure()
        fig.update_layout(scene=scene, title="Point cloud")

        fig.add_trace(gobj.Scatter3d(x=point_cloud[:, 0],
                                     y=point_cloud[:, 1],
                                     z=point_cloud[:, 2],
                                     mode='markers',
                                     marker=dict(size=4,
                                                 color=list(range(
                                                     point_cloud.shape[0])),
                                                 colorscale='Viridis',
                                                 opacity=0.8)))

        fig.show()
    else:
        raise ValueError("The value of the dimension is different from 2 or 3")

                      
def plot_diagram(diagram, homology_dimensions=None):
    """Plot a single persistence diagram.

    Parameters
    ----------
    diagram : ndarray, shape (n_points, 3)
        The persistence diagram to plot, where the third dimension along axis 1
        contains homology dimensions, and the other two contain (birth, death)
        pairs to be used as coordinates in the two-dimensional plot.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions which will appear on the plot. If ``None``, all
        homology dimensions which appear in `diagram` will be plotted.

    """
    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])
    
    maximum_persistence = np.where(np.isinf(diagram), -np.inf, diagram).max()

    layout = {
        "title": "Persistence diagram", 
        "width": 500,
        "height": 500,
        "xaxis1": {
            "title": "Birth",
            "side": "bottom", 
            "type": "linear", 
            "range": [0, 1.1 * maximum_persistence],
            "ticks": "outside", 
            "anchor": "y1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        }, 
        "yaxis1": {
            "title": "Death",
            "side": "left", 
            "type": "linear", 
            "range": [0, 1.1 * maximum_persistence],
            "ticks": "outside", 
            "anchor": "x1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        }, 
        "plot_bgcolor": "white"
    }

    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)

    fig.add_trace(gobj.Scatter(x=np.array([-100 * maximum_persistence,
                                           100 * maximum_persistence]),
                               y=np.array([-100 * maximum_persistence,
                                           100 * maximum_persistence]),
                               mode='lines',
                               line=dict(dash='dash', width=1, color='black'),
                               showlegend=False, hoverinfo='none'))
    
    for i, dimension in enumerate(homology_dimensions):
        name = "H{}".format(int(dimension))
        subdiagram = _subdiagrams(np.asarray([diagram]), [dimension],
                                  remove_dim=True)[0]
        diff = (subdiagram[:, 1] != subdiagram[:, 0])
        subdiagram = subdiagram[diff]
        fig.add_trace(gobj.Scatter(x=subdiagram[:, 0], y=subdiagram[:, 1],
                                   mode='markers', name=name))

    fig.show()


def plot_landscapes(landscapes, homology_dimensions=None, samplings=None):
    """Plot landscapes by homology dimension.

    Parameters
    ----------
    landscapes : ndarray, shape (n_homology_dimension, n_layers, n_values)
        Collection of ``n_homology_dimension`` discretised persistence
        landscapes. Each landscape contains ``n_layers`` layers. Entry i along
        axis 0 should be the persistence landscape in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti curves should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray, shape (n_homology_dimension, n_layers, n_values), \
                default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `landscapes` on
        the y-axis. If ``None``, the samplings will start at 0 with step 1.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, landscapes.shape[0])
    if samplings is None:
        samplings = np.arange(0, landscapes.shape[2])
    layout = {
        "xaxis1": {
            "side": "bottom", 
            "type": "linear", 
            "ticks": "outside", 
            "anchor": "y1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        }, 
        "yaxis1": {
            "side": "left", 
            "type": "linear", 
            "ticks": "outside", 
            "anchor": "x1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        }, 
        "plot_bgcolor": "white"
    }
    
    for i, dimension in enumerate(homology_dimensions):
        layout_dim = layout.copy()
        layout_dim['title'] = "Persistence landscape for homology dimension" + \
                              "{}".format(int(dimension))
        fig = gobj.Figure(layout=layout_dim)
        fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                         mirror=False)
        fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                         mirror=False)

        n_layers = landscapes.shape[1]
        for layer in range(n_layers):
            fig.add_trace(gobj.Scatter(x=samplings,
                                       y=landscapes[i, layer, :],
                                       mode='lines', showlegend=False,
                                       hoverinfo='none',
                                       name="layer {}".format(layer + 1)))

        fig.show()


def plot_betti_curves(betti_curves, homology_dimensions=None, samplings=None):
    """Plot the Betti curves of a single persistence diagram by homology
    dimension.
        
    Parameters
    ----------
    betti_curves : ndarray, shape (n_homology_dimension, n_values)
        Collection of ``n_homology_dimension`` discretised Betti curves.
        Entry i along axis 0 should be the Betti curve in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti curves should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray, shape (n_homology_dimension, n_values), \
                default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `betti_curves` on
        the y-axis. If ``None``, the samplings will start at 0 with step 1.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, betti_curves.shape[0])
    if samplings is None:
        samplings = np.arange(0, betti_curves.shape[1])
    layout = {
        "title": "Betti curves",
        "xaxis1": {
            "title": "Epsilon",
            "side": "bottom",
            "type": "linear",
            "ticks": "outside",
            "anchor": "x1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "yaxis1": {
            "title": "Betti number",
            "side": "left",
            "type": "linear",
            "ticks": "outside",
            "anchor": "y1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "plot_bgcolor": "white"
    }
    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)

    for i, dimension in enumerate(homology_dimensions):
        fig.add_trace(gobj.Scatter(x=samplings,
                                   y=betti_curves[i, :],
                                   mode='lines', showlegend=False,
                                   hoverinfo='none'))
        
    fig.show()


def plot_betti_surfaces(betti_curves, samplings=None,
                        homology_dimensions=None):
    """Plots the Betti surfaces (Betti numbers against time and filtration
    parameter) by homology dimension.

    Parameters
    ----------
    betti_curves : ndarray, shape (n_samples, n_homology_dimensions, \
                   n_values)
        ``n_samples`` collections of discretised Betti curves. There are
        ``n_homology_dimension`` curves in each collection. Index i along axis
        1 should yield all Betti curves in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti surfaces should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray, shape (n_homology_dimension, n_values), \
                default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `betti_curves` on the
        y-axis. If ``None``, the samplings will start at 0 with step 1.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, betti_curves.shape[1])
    if samplings is None:
        samplings = np.arange(0, betti_curves.shape[2])

    scene = {
        "xaxis": {
            "title": "Epsilon",
            "type": "linear",
            "showexponent": "all",
            "exponentformat": "e"
        }, 
        "yaxis": {
            "title": "Time",
            "type": "linear", 
            "showexponent": "all",
            "exponentformat": "e"
        },
        "zaxis": {
            "title": "Betti number",
            "type": "linear", 
            "showexponent": "all",
            "exponentformat": "e"
        } 
    }
    if betti_curves.shape[0] == 1:
        plot_betti_curves(betti_curves[0], samplings, homology_dimensions)
    else:
        for i, dimension in enumerate(homology_dimensions):
            fig = gobj.Figure()
            fig.update_layout(scene=scene, title="Betti surface for homology "
                                                 "dimension {}".format(int(dimension)))
            fig.add_trace(gobj.Surface(x=samplings,
                                       y=np.arange(betti_curves.shape[0]),
                                       z=betti_curves[:, i, :],
                                       connectgaps=True, hoverinfo='none'))
            
            fig.show()
