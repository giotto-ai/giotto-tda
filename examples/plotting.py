"""Plot functions """

import numpy as np
import plotly.graph_objs as gobj


def plot_diagram(diagram, homology_dimensions=None):
    """Plots one persistence diagram.

    Parameters
    ----------
    diagram : dict of np.arrays of shape (*,2)
        The persistence diagram to plot, where the keys of the dict correspond to homology dimensions and each
        entry is the collection of (birth,death) points in R^2 of the corresponding homology dimension.
    homology_dimensions : list of ints, default None
        The list of homology dimensions that will appear on the plot. None means that all the homology dimensions
        contained in diagram will be plotted.
    """
    if homology_dimensions is None:
        homology_dimensions = diagram.keys()

    maximum_persistence = 0
    for sub_diagram in diagram.values():
        if np.max(sub_diagram) > maximum_persistence:
            maximum_persistence = np.max(sub_diagram)

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
    
    for dimension in homology_dimensions:
        name = 'H'+str(dimension)
        fig.add_trace(gobj.Scatter(x=diagram[dimension][:,0], y=diagram[dimension][:,1], mode='markers', name=name))

    fig.show()


def plot_landscapes(landscape, homology_dimensions=None):
    """Plots the landscapes by homology dimension.

    Parameters
    ----------
    landscape : dict of np.arrays of shape (n_layers,n_sampled_values,2)
        The landcsapes to plot, where the keys of the dict correspond to homology dimensions.
    homology_dimensions : list of ints, default None
        The list of homology dimensions for which the landscape is plotted. None means that the landscape of
        every homology dimension is plotted.
    """
    if homology_dimensions is None:
        homology_dimensions = landscape.keys()

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
    
    for dimension in homology_dimensions:
        layout_dim = layout.copy()
        layout_dim['title'] = "Persistence landscape for homology dimension "+str(dimension)
        fig = gobj.Figure(layout=layout_dim)
        fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)
        fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)

        n_layers = landscape[dimension].shape[0]
        for layer in range(n_layers):
            fig.add_trace(gobj.Scatter(x=landscape[dimension][layer,:,0], y=landscape[dimension][layer,:,1],
            mode='lines', showlegend=False, hoverinfo='none', name='layer '+str(layer+1)))

        fig.show()


def plot_betti_curves(betti_curves, homology_dimensions=None):
    """ Plots the Betti curves of different homology dimensions on the same
    plot.

    Parameters
    ----------
    betti_curves : dict of np.arrays of shape (n_sampled_values,2)
        The Betti curves to plot, where the keys of the dict correspond to homology dimensions.
    homology_dimensions : list of ints, default None
        The list of homology dimensions that will appear on the plot. None means that the Betti curves of every homology
        dimension contained in betti_curves will be plotted.
    """
    if homology_dimensions is None:
        homology_dimensions = betti_curves.keys()

    min_sampled_value = np.inf
    max_sampled_value = 0
    for betti_curve in betti_curves.values():
        if np.max(betti_curve[:,0]) > max_sampled_value:
            max_sampled_value = np.max(betti_curve[:,0])
        if np.min(betti_curve[:,0]) < min_sampled_value:
            min_sampled_value = np.min(betti_curve[:,0])
    
    layout = {
        "title": "Betti curves", 
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

    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black', mirror=False)
    
    for dimension in homology_dimensions:
        name = 'H'+str(dimension)

        betti_curve = betti_curves[dimension].copy()
        if dimension==0:
            betti_curve = np.vstack([betti_curve, np.array([max_sampled_value, 1])])
        else:
            betti_curve = np.vstack([np.array([min_sampled_value,0]), betti_curve, np.array([max_sampled_value,0])])
        
        fig.add_trace(gobj.Scatter(x=betti_curve[:,0], y=betti_curve[:,1], mode='lines', hoverinfo='none', name=name))
    
    fig.show()


def plot_betti_surfaces(betti_curves, homology_dimensions=None):
    """
    Plots the Betti surfaces (Betti number against time and epsilon) by homology dimension.

    Parameters
    ----------
    betti_curves : collection (list or np.array) of dict of np.arrays of shape (n_sampled_values,2)
        The Betti curves across time , where the keys of the dict correspond to homology dimensions.
    homology_dimensions : list of ints, default None
        The list of homology dimensions for which the Betti surface is plotted. None means that the Betti surface of
        every homology dimension is plotted.
    """
    if homology_dimensions is None:
        homology_dimensions = betti_curves[0].keys()

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

    for dimension in homology_dimensions:
        fig = gobj.Figure()
        fig.update_layout(scene=scene, title="Betti surface for homology dimension "+str(dimension))

        betti_surface = np.array([betti_curves[i][dimension] for i in range(len(betti_curves))])
        sampling = betti_surface[0,:,0]

        fig.add_trace(gobj.Surface(x=sampling, y=np.arange(betti_surface.shape[0]), z=betti_surface[:,:,1],
        connectgaps=True, hoverinfo='none'))
        
        fig.show()
        




