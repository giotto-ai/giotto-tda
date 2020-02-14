"""Static and interactive visualisation functions for Mapper graphs."""
# License: GNU AGPLv3

import logging
import traceback

import numpy as np
import plotly.graph_objects as go
from ipywidgets import Layout, widgets
from sklearn.base import clone

from .utils._logging import OutputWidgetHandler
from .utils.visualization import (_calculate_graph_data,
                                  _get_column_color_buttons)


def plot_static_mapper_graph(
        pipeline, data, layout='kamada_kawai', layout_dim=2,
        color_variable=None, node_color_statistic=None,
        color_by_columns_dropdown=False, plotly_kwargs=None,
        clone_pipeline=True):
    """Plotting function for static Mapper graphs.

    Nodes are colored according to :attr:`color_variable`. By default, the
    hovertext displays a globally unique ID and the number of elements
    associated with a given node.

    Parameters
    ----------
    pipeline : :class:`~gtda.mapper.pipeline.MapperPipeline` object
        Mapper pipeline to act on to data.

    data : array-like of shape (n_samples, n_features)
        Data used to generate the Mapper graph. Can be a pandas dataframe.

    layout : None, str or callable, optional, default: ``'kamada-kawai'``
        Layout algorithm for the graph. Can be any accepted value for the
        ``layout`` parameter in the :meth:`layout` method of
        :class:`igraph.Graph`. [1]_

    layout_dim : int, default: ``2``
        The number of dimensions for the layout. Can be 2 or 3.

    color_variable : object or None, optional, default: ``None``
        Specifies which quantity is to be used for node coloring.

            1. If a numpy ndarray or pandas dataframe, `color_variable`
               must have the same length as `data` and is interpreted as
               a quantity of interest according to which node of the Mapper
               graph is to be colored (see `node_color_statistic`).
            2. If ``None`` then equivalent to passing `data`.
            3. If an object implementing :meth:`transform` or
               :meth:`fit_transform`, e.g. a scikit-learn estimator or
               pipeline, it is applied to `data` to generate the quantity
               of interest.
            4. If an index or string, or list of indices / strings, equivalent
               to selecting a column or subset of columns from `data`.

    node_color_statistic : None, callable, or ndarray of shape (n_nodes,) or \
        (n_nodes, 1), optional, default: ``None``
        Specifies how to determine the colors of each node. If a
        numpy array, it must have the same length as the number of nodes in
        the Mapper graph, and its values are used directly for node
        coloring, ignoring `color_variable`. Otherwise, it can be a
        callable object which is used to obtain a summary statistic, within
        each Mapper node, of the quantity specified by `color_variable`. The
        default value ``None`` is equivalent to passing ``numpy.mean``.

    color_by_columns_dropdown : bool, optional, default: ``False``
        If ``True``, a dropdown widget is generated which allows the user to
        color Mapper nodes according to any column in `data`.

    plotly_kwargs : dict, optional, default: ``None``
        Keyword arguments to configure the Plotly Figure.

    clone_pipeline : bool, optional, default: ``True``
        If ``True``, the input `pipeline` is cloned before computing the
        Mapper graph to prevent unexpected side effects from in-place
        parameter updates.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing the Mapper graph with appropriate node colouring
        and size.

    References
    ----------
    .. [1] `igraph.Graph.layout
            <https://igraph.org/python/doc/igraph.Graph-class.html#layout>`_
            documentation.

    """

    # Compute the graph and fetch the indices of points in each node
    if clone_pipeline:
        pipe = clone(pipeline)
    else:
        pipe = pipeline

    if node_color_statistic is not None:
        _node_color_statistic = node_color_statistic
    else:
        _node_color_statistic = np.mean

    # Simple duck typing to determine whether data is a pandas dataframe
    is_data_dataframe = hasattr(data, 'columns')

    node_trace, edge_trace, node_elements, _node_colors, plot_options = \
        _calculate_graph_data(
            pipe, data, layout, layout_dim,
            color_variable, _node_color_statistic, plotly_kwargs)

    # Define layout options that are common to 2D and 3D figures
    layout_options_common = go.Layout(
        showlegend=plot_options['layout_showlegend'],
        hovermode=plot_options['layout_hovermode'],
        margin=plot_options['layout_margin'],
        autosize=False
    )

    fig = go.FigureWidget(data=[edge_trace, node_trace],
                          layout=layout_options_common)

    if layout_dim == 2:
        layout_options_2d = {
            'layout_xaxis': plot_options['layout_xaxis'],
            'layout_xaxis_title': plot_options['layout_xaxis_title'],
            'layout_yaxis': plot_options['layout_yaxis'],
            'layout_yaxis_title': plot_options['layout_yaxis_title'],
            'layout_template': 'simple_white',
        }
        fig.update(layout_options_2d)

    elif layout_dim == 3:
        layout_options_3d = {
            'layout_scene': plot_options['layout_scene'],
            'layout_annotations': plot_options['layout_annotations'],
        }
        fig.update(layout_options_3d)

    # Compute node colors according to data columns only if necessary
    if color_by_columns_dropdown:
        column_color_buttons = _get_column_color_buttons(
            data, is_data_dataframe, node_elements, _node_colors,
            plot_options['node_trace_marker_colorscale'])
    else:
        column_color_buttons = None

    button_height = 1.1
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=column_color_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor='left',
                y=button_height,
                yanchor="top"
            ),
        ])

    if color_by_columns_dropdown:
        fig.add_annotation(
            go.layout.Annotation(text="Color by:", x=0, xref="paper",
                                 y=button_height - 0.045,
                                 yref="paper", align="left", showarrow=False)
        )

    return fig


def plot_interactive_mapper_graph(pipeline, data, layout='kamada_kawai',
                                  layout_dim=2, color_variable=None,
                                  node_color_statistic=None,
                                  color_by_columns_dropdown=False,
                                  plotly_kwargs=None):
    """Plotting function for interactive Mapper graphs.

    Provides functionality to interactively update parameters from the cover
    and clustering steps defined in :attr:`pipeline`. Nodes are colored
    according to :attr:`color_variable`. By default, the hovertext displays a
    globally unique ID and the number of elements associated with a given node.

    Parameters
    ----------
    pipeline : :class:`~gtda.mapper.pipeline.MapperPipeline` object
        Mapper pipeline to act on to data.

    data : array-like of shape (n_samples, n_features)
        Data used to generate the Mapper graph. Can be a pandas dataframe.

    layout : None, str or callable, optional, default: ``'kamada-kawai'``
        Layout algorithm for the graph. Can be any accepted value for the
        ``layout`` parameter in the :meth:`layout` method of
        :class:`igraph.Graph`. [1]_

    layout_dim : int, default: ``2``
        The number of dimensions for the layout. Can be 2 or 3.

    color_variable : object or None, optional, default: ``None``
        Specifies which quantity is to be used for node coloring.

            1. If a numpy ndarray or pandas dataframe, `color_variable`
               must have the same length as `data` and is interpreted as
               a quantity of interest according to which node of the Mapper
               graph is to be colored (see `node_color_statistic`).
            2. If ``None`` then equivalent to passing `data`.
            3. If an object implementing :meth:`transform` or
               :meth:`fit_transform`, e.g. a scikit-learn estimator or
               pipeline, it is applied to `data` to generate the quantity
               of interest.
            4. If an index or string, or list of indices / strings, equivalent
               to selecting a column or subset of columns from `data`.

    node_color_statistic :None, callable, or ndarray of shape (n_nodes,) or \
        (n_nodes, 1), optional, default: ``None``
        Specifies how to determine the colors of each node. If a
        numpy array, it must have the same length as the number of nodes in
        the Mapper graph, and its values are used directly for node
        coloring, ignoring `color_variable`. Otherwise, it can be a
        callable object which is used to obtain a summary statistic, within
        each Mapper node, of the quantity specified by `color_variable`. The
        default value ``None`` is equivalent to passing ``numpy.mean``.

    color_by_columns_dropdown : bool, optional, default: ``False``
        If ``True``, a dropdown widget is generated which allows the user to
        color Mapper nodes according to any column in `data`.

    plotly_kwargs : dict, optional, default: ``None``
        Keyword arguments to configure the Plotly Figure.

    Returns
    -------
    box : :class:`ipywidgets.VBox` object
    A box containing the following widgets: parameters of the clustering
    algorithm, parameters for the covering scheme, a Mapper graph arising
    from those parameters, a validation box, and logs.

    References
    ----------
    .. [1] `igraph.Graph.layout
            <https://igraph.org/python/doc/igraph.Graph-class.html#layout>`_
            documentation.

    """

    # clone pipeline to avoid side effects from in-place parameter changes
    pipe = clone(pipeline)

    if node_color_statistic is not None:
        _node_color_statistic = node_color_statistic
    else:
        _node_color_statistic = np.mean

    def get_widgets_per_param(param, value):
        if isinstance(value, float):
            return (param, widgets.FloatText(
                value=value,
                step=0.05,
                description=param.split('__')[1],
                continuous_update=False,
                disabled=False
            ))
        elif isinstance(value, int):
            return (param, widgets.IntText(
                value=value,
                step=1,
                description=param.split('__')[1],
                continuous_update=False,
                disabled=False
            ))
        elif isinstance(value, str):
            return (param, widgets.Text(
                value=value,
                description=param.split('__')[1],
                continuous_update=False,
                disabled=False
            ))
        else:
            return None

    def update_figure(figure, edge_trace, node_trace, layout_dim):
        figure.data[0].x = edge_trace.x
        figure.data[0].y = edge_trace.y
        figure.data[1].x = node_trace.x
        figure.data[1].y = node_trace.y

        if layout_dim == 3:
            figure.data[0].z = edge_trace.z
            figure.data[1].z = node_trace.z

        figure.data[1].marker.size = node_trace.marker.size
        figure.data[1].marker.color = node_trace.marker.color
        figure.data[1].marker.cmin = node_trace.marker.cmin
        figure.data[1].marker.cmax = node_trace.marker.cmax
        figure.data[1].marker.sizeref = node_trace.marker.sizeref
        figure.data[1].hoverlabel = node_trace.hoverlabel
        figure.data[1].hovertext = node_trace.hovertext

    def on_parameter_change(change):
        handler.clear_logs()
        try:
            for param, value in cover_params.items():
                if isinstance(value, (int, float, str)):
                    pipe.set_params(
                        **{param: cover_params_widgets[param].value})
            for param, value in cluster_params.items():
                if isinstance(value, (int, float, str)):
                    pipe.set_params(
                        **{param: cluster_params_widgets[param].value})

            logger.info("Updating figure ...")
            with fig.batch_update():
                (node_trace, edge_trace, node_elements, node_colors,
                 plot_options) = _calculate_graph_data(
                    pipe, data, layout, layout_dim,
                    color_variable, _node_color_statistic, plotly_kwargs
                )
                update_figure(fig, edge_trace, node_trace, layout_dim)

                # Update color by column buttons
                is_data_dataframe = hasattr(data, 'columns')
                if color_by_columns_dropdown:
                    column_color_buttons = _get_column_color_buttons(
                        data, is_data_dataframe, node_elements, node_colors,
                        plot_options['node_trace_marker_colorscale'])
                else:
                    column_color_buttons = None

                button_height = 1.1
                fig.update_layout(
                    updatemenus=[
                        go.layout.Updatemenu(
                            buttons=column_color_buttons,
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.11,
                            xanchor='left',
                            y=button_height,
                            yanchor="top"
                        ),
                    ])

            valid.value = True
        except Exception:
            exception_data = traceback.format_exc().splitlines()
            logger.exception(exception_data[-1])
            valid.value = False

    def observe_widgets(params, widgets):
        for param, value in params.items():
            if isinstance(value, (int, float, str)):
                widgets[param].observe(on_parameter_change, names='value')

    # define output widget to capture logs
    out = widgets.Output()

    @out.capture()
    def click_box(change):
        if logs_box.value:
            out.clear_output()
            handler.show_logs()
        else:
            out.clear_output()

    # initialise logging
    logger = logging.getLogger(__name__)
    handler = OutputWidgetHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # initialise cover and cluster dictionaries of parameters and widgets
    cover_params = dict(filter(lambda x: x[0].startswith('cover'),
                               pipe.get_mapper_params().items()))
    cover_params_widgets = dict(
        filter(None, map(lambda x: get_widgets_per_param(*x),
                         cover_params.items())))
    cluster_params = dict(filter(lambda x: x[0].startswith('clusterer'),
                                 pipe.get_mapper_params().items()))
    cluster_params_widgets = dict(
        filter(None, map(lambda x: get_widgets_per_param(*x),
                         cluster_params.items())))

    # initialise widgets for validating input parameters of pipeline
    valid = widgets.Valid(
        value=True,
        description='Valid parameters',
        style={'description_width': '100px'},
    )

    # initialise widget for showing the logs
    logs_box = widgets.Checkbox(
        description='Show logs: ',
        value=False,
        indent=False
    )

    # initialise figure with initial pipeline and config
    if plotly_kwargs is None:
        plotly_kwargs = dict()

    fig = plot_static_mapper_graph(
        pipe, data, layout, layout_dim, color_variable, _node_color_statistic,
        color_by_columns_dropdown, plotly_kwargs, clone_pipeline=False)

    observe_widgets(cover_params, cover_params_widgets)
    observe_widgets(cluster_params, cluster_params_widgets)

    logs_box.observe(click_box, names='value')

    # define containers for input widgets
    container_cover = widgets.HBox(
        children=list(cover_params_widgets.values()))

    container_cluster_layout = Layout(display='flex', flex_flow='row wrap')

    container_cluster = widgets.HBox(
        children=list(cluster_params_widgets.values()),
        layout=container_cluster_layout)

    box = widgets.VBox(
        [container_cover, container_cluster, fig, valid, logs_box, out])
    return box
