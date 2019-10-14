"""Plot functions """

import numpy as np
import plotly.graph_objs as gobj
from giotto.diagram._utils import _subdiagrams

def plot_point_cloud(point_cloud, dimension = None):
    """This functions plot the first 2 or 3 coordinates of the point cloud.
    This function will not work for 1-dimensional arrays
        
    Parameters
    ----------
    dimension : int , default : ``None``
        This parameter sets the dimension of the resulting plot. If ``None``, the
        dimension will be chosen between 2 and 3 depending on `n_features` (see
        Input).
        
    Input
    -----
    point_cloud : ndarray of shape (n_samples, n_features)
        the point cloud is the set of data points to be rapresented in a 2D or 3D
        scatter plot. Only the first 2 or 3 dimensions will be consdered for plotting.
    """
    if dimension is None:
        dimension = np.min((3, point_cloud.shape[1]))

    # Check consistency between point_cloud and dimension
    if point_cloud.shape[1] < dimension :
        raise ValueError("The `n_features` of the point cloud ìs less than the `dimension`")

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
                "exponentformat" : "e"
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
                "exponentformat" : "e"
            },
            "plot_bgcolor": "white"
        }
    
        fig = gobj.Figure(layout=layout)
        fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)
        fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)

        fig.add_trace(gobj.Scatter(x=point_cloud[:,0],
                                   y=point_cloud[:,1],
                                   mode='markers',
                                   marker=dict(size=4,
                                               color=list(range(point_cloud.shape[0])),
                                               colorscale='Viridis', opacity=0.8)))
        fig.show()
    elif dimension == 3:
        
        scene = {
            "xaxis": {
                "title": "First coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat" : "e"
            },
            "yaxis": {
                "title": "Second coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat" : "e"
            },
            "zaxis": {
                "title": "Third coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat" : "e"
            }
        }

        fig = gobj.Figure()
        fig.update_layout(scene=scene, title="Point Cloud")

        fig.add_trace(gobj.Scatter3d(x=point_cloud[:,0],
                                     y=point_cloud[:,1],
                                     z=point_cloud[:,2],
                                     mode='markers',
                                     marker=dict(size=4,
                                                 color=list(range(point_cloud.shape[0])),
                                                 colorscale='Viridis', opacity=0.8)))

        fig.show()
    else:
        raise ValueError("The value of the parameter ´dimension´ is different from 2 or 3")

                      
def plot_diagram(diagram, homology_dimensions=None):
    """Plots one persistence diagram.

    Parameters
    ----------
    homology_dimensions : list of int, default: ``None``
        The list of homology dimensions that will appear on the plot. None means that all the homology dimensions
        contained in diagram will be plotted.
        
    Input
    -----
    diagram : ndarray of shape (n_points, 3)
    The persistence diagram to plot, where the keys of the dict correspond to homology dimensions and each
    entry is the collection of (birth,death) points in R^2 of the corresponding homology dimension.
    """
    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:,2])
    
    maximum_persistence = np.where(np.isinf(diagram),-np.inf,diagram).max()

    layout = {
        "title": "Persistence diagram", 
        "width": 500,
        "height": 500,
        "xaxis1": {
            "title": "Birth",
            "side": "bottom", 
            "type": "linear", 
            "range": [0, 1.1*maximum_persistence], 
            "ticks": "outside", 
            "anchor": "y1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat" : "e"
        }, 
        "yaxis1": {
            "title": "Death",
            "side": "left", 
            "type": "linear", 
            "range": [0, 1.1*maximum_persistence], 
            "ticks": "outside", 
            "anchor": "x1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat" : "e"
        }, 
        "plot_bgcolor": "white"
    }

    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)

    fig.add_trace(gobj.Scatter(x=np.array([-100*maximum_persistence,100*maximum_persistence]),
    y=np.array([-100*maximum_persistence,100*maximum_persistence]), mode='lines',
    line=dict(dash='dash',width=1,color='black'), showlegend=False, hoverinfo='none'))
    
    for i, dimension in enumerate(homology_dimensions):
        name = 'H'+str(int(dimension))
        subdiagram = _subdiagrams(np.asarray([diagram]),[dimension],remove_dim=True)[0]
        diff = (subdiagram[:, 1] != subdiagram[:, 0])
        subdiagram = subdiagram[diff]
        fig.add_trace(gobj.Scatter(x=subdiagram[:,0], y=subdiagram[:,1], mode='markers', name=name))

    fig.show()


def plot_landscapes(landscape, samplings=None, homology_dimensions=None):
    """Plots the landscapes by homology dimension.

    Parameters
    ----------
    homology_dimensions : list of int, default: ``None``
        The list of the homology group's ranks of which one wants to plot the landscape.
        If no list of dimensions is passed, we assume that they start from H0 with step 1.
        
    Input
    -----
    landscape : ndarray of shape (n_homology_dimension, n_layers, n_values)
        The values of the persistence landsacpe. ``n_homology_dimension`` is the length
        of the ``homology_dimensions`` array; ``n_layers`` is the number of landscape
        profiles and ``n_values``is the number of samples.
    samplings : ndarray of shape (n_homology_dimension, n_layers, n_values), default: ```None``
        The x axis of the persistence landscape. ``n_homology_dimension`` is the length
        of the ``homology_dimensions`` array; ``n_layers`` is the number of landscape
        profiles and ``n_values``is the number of samples.`If no value is input, the
        samplings will start at 0 with step 1.
    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0,landscape.shape[0])
    if samplings is None:
        samplings = np.arange(0,landscape.shape[2])
    layout = {
        "xaxis1": {
            "side": "bottom", 
            "type": "linear", 
            "ticks": "outside", 
            "anchor": "y1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat" : "e"
        }, 
        "yaxis1": {
            "side": "left", 
            "type": "linear", 
            "ticks": "outside", 
            "anchor": "x1",  
            "showline": True, 
            "zeroline": True,
            "showexponent": "all",
            "exponentformat" : "e"
        }, 
        "plot_bgcolor": "white"
    }
    
    for i, dimension in enumerate(homology_dimensions):
        layout_dim = layout.copy()
        layout_dim['title'] = "Persistence landscape for homology dimension "+str(int(dimension))
        fig = gobj.Figure(layout=layout_dim)
        fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)
        fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)

        n_layers = landscape.shape[1]
        for layer in range(n_layers):
            fig.add_trace(gobj.Scatter(x=samplings,
                                       y=landscape[i,layer,:],
            mode='lines', showlegend=False, hoverinfo='none', name='layer '+str(layer+1)))

        fig.show()


def plot_betti_curves(betti_curves, samplings=None, homology_dimensions=None):
    """Plots the landscapes by homology dimension.
        
    Parameters
    ----------
    homology_dimensions : list of int, default: ``None``
        The list of the homology group's ranks of which one wants to plot the landscape.
        If no list of dimensions is passed, we assume that they start from H0 with step 1.
        
    Input
    -----
    betti_curves : ndarray of shape (n_homology_dimension, n_values)
        The values of the Betti curves. ``n_homology_dimension`` is the length
        of the ``homology_dimensions`` array and ``n_values``is the number of samples.
    samplings : ndarray of shape (n_homology_dimension, n_values), default: ```None``
        The x axis of the Betti curves. ``n_homology_dimension`` is the length
        of the ``homology_dimensions`` array and ``n_values``is the number of samples.`If no value
        is input, the samplings will start at 0 with step 1.
    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0,betti_curves.shape[0])
    if samplings is None:
        samplings = np.arange(0,betti_curves.shape[1])
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
            "exponentformat" : "e"
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
            "exponentformat" : "e"
        },
        "plot_bgcolor": "white"
    }
    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)

    for i, dimension in enumerate(homology_dimensions):
        fig.add_trace(gobj.Scatter(x=samplings,
                                   y=betti_curves[i,:],
                                   mode='lines', showlegend=False,
                                   hoverinfo='none'))
        
    fig.show()


def plot_betti_surfaces(betti_curves, samplings=None, homology_dimensions=None):
    """
    Plots the Betti surfaces (Betti number against time and epsilon) by homology dimension.

    Parameters
    ----------
    betti_curves : ndarray of shape (n_samples, n_homology_dimensions, n_values)
        The Betti curves across time, sampled in ``n_samples`` samples.
        ``n_homology_dimension`` is the length
        of the ``homology_dimensions`` array and ``n_values``is the number of samples.
    homology_dimensions : list of ints, default ``None``
        The list of homology dimensions for which the Betti surface is plotted.
        None means that the Betti surface of
        every homology dimension is plotted.
    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0,betti_curves.shape[1])
    if samplings is None:
        samplings = np.arange(0,betti_curves.shape[2])

    scene = {
        "xaxis": {
            "title": "Epsilon",
            "type": "linear",
            "showexponent": "all",
            "exponentformat" : "e"
        }, 
        "yaxis": {
            "title": "Time",
            "type": "linear", 
            "showexponent": "all",
            "exponentformat" : "e"
        },
        "zaxis": {
            "title": "Betti number",
            "type": "linear", 
            "showexponent": "all",
            "exponentformat" : "e"
        } 
    }

    for i, dimension in enumerate(homology_dimensions):
        fig = gobj.Figure()
        fig.update_layout(scene=scene, title="Betti surface for homology dimension "+str(dimension))
        fig.add_trace(gobj.Surface(x=samplings, y=np.arange(betti_curves.shape[0]),
                                   z=betti_curves[:,i,:],connectgaps=True, hoverinfo='none'))
        
        fig.show()
        




