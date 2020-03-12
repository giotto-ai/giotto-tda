"""Graph layout functions and plotly layout functions."""
# License: GNU AGPLv3

import operator
from functools import reduce

import numpy as np
import plotly.graph_objs as go
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex


def _get_node_size(node_elements):
    # TODO: Add doc strings to all functions
    return list(map(len, node_elements))


def _get_node_text(graph):
    return [
        f"Node ID: {node_id}<br>Node size: {len(node_elements)}"
        for node_id, node_elements in zip(
            graph["node_metadata"]["node_id"],
            graph["node_metadata"]["node_elements"])
        ]


def set_node_sizeref(node_elements, node_scale=12):
    # Formula from Plotly https://plot.ly/python/bubble-charts/
    return 2. * max(_get_node_size(node_elements)) / (node_scale ** 2)


def get_node_summary(node_elements, data, summary_stat=np.mean):
    return np.asarray(
        list(map(summary_stat, [data[itr] for itr in node_elements]))
    )


def _get_column_color_buttons(data, is_data_dataframe, node_elements,
                              node_colors_color_variable, colorscale):
    # TODO: Consider opting for just-in-time computation instead of computing
    # all node summary values ahead-of-time. Solution should preserve scroll
    # zoom functionality of 2D static visualisation.
    if is_data_dataframe:
        columns_to_color = data.columns
    else:
        columns_to_color = range(data.shape[1])

    node_color_map = list(map(lambda x: rgb2hex(get_cmap(colorscale)(x)),
                              node_colors_color_variable))

    column_color_buttons = [
        dict(
            args=[{
                'marker.color': [None, node_color_map],
                'marker.cmin': [None, np.min(node_colors_color_variable)],
                'marker.cmax': [None, np.max(node_colors_color_variable)],
                'hoverlabel.bgcolor': [None, node_color_map]
            }],
            label='color_variable',
            method='restyle'
        )
    ]

    for column in columns_to_color:
        if is_data_dataframe:
            column_values = data[column].to_numpy()
        else:
            column_values = data[:, column]
        node_colors = get_node_summary(node_elements, column_values)
        node_color_map = list(map(lambda x: rgb2hex(get_cmap(colorscale)(x)),
                                  node_colors))

        column_color_buttons.append(
            dict(
                args=[{
                    'marker.color': [None, node_color_map],
                    'marker.cmin': [None, np.min(node_colors)],
                    'marker.cmax': [None, np.max(node_colors)],
                    'hoverlabel.bgcolor': [None, node_color_map]
                }],
                label=f'Column {column}',
                method='restyle'
            )
        )
    return column_color_buttons


def _infer_color_variable_kind(color_variable, data):
    """Determines whether color_variable is array, pandas dataframe, callable,
    or scikit-learn transformer or fit_transformer."""
    if hasattr(color_variable, 'dtype') or hasattr(color_variable, 'dtypes'):
        if len(color_variable) != len(data):
            raise ValueError(
                "color_variable and data must have the same length.")
        color_variable_kind = 'scalars'
    elif hasattr(color_variable, 'transform'):
        color_variable_kind = 'transformer'
    elif hasattr(color_variable, 'fit_transform'):
        color_variable_kind = 'fit_transformer'
    elif callable(color_variable):
        color_variable_kind = 'callable'
    elif color_variable is None:
        color_variable_kind = 'none'
    else:  # Assume color_variable is a selection of columns
        color_variable_kind = 'else'

    return color_variable_kind


def _get_node_colors(data, is_data_dataframe, node_elements,
                     is_node_colors_ndarray, node_color_statistic,
                     color_variable, color_variable_kind):
    """Calculate node colors"""
    if is_node_colors_ndarray:
        node_colors = node_color_statistic
    else:
        if color_variable_kind == 'scalars':
            color_data = color_variable
        elif color_variable_kind == 'transformer':
            color_data = color_variable.transform(data)
        elif color_variable_kind == 'fit_transformer':
            color_data = color_variable.fit_transform(data)
        elif color_variable_kind == 'callable':
            color_data = color_variable(data)
        elif color_variable_kind == 'none':
            if is_data_dataframe:
                color_data = data.to_numpy()
            else:
                color_data = data
        else:
            if is_data_dataframe:
                color_data = data[color_variable].to_numpy()
            else:
                color_data = data[:, color_variable]

        node_colors = get_node_summary(
            node_elements, color_data, summary_stat=node_color_statistic)

    # Check if node_colors contains NaNs
    if any(np.logical_not(np.isfinite(node_colors))):
        from warnings import warn
        warn('NaN values detected in the array of Mapper node colors!'
             'These values will be ignored in the color scale', RuntimeWarning)

    # Normalise node colours in range [0,1] for colorscale mapping
    node_colors = (node_colors - np.nanmin(node_colors)) / \
        (np.nanmax(node_colors) - np.nanmin(node_colors))

    return node_colors


def _calculate_graph_data(
        pipeline, data, layout, layout_dim,
        color_variable, node_color_statistic, plotly_kwargs):
    graph = pipeline.fit_transform(data)
    node_elements = graph['node_metadata']['node_elements']

    # Simple duck typing to determine whether data is a pandas dataframe
    is_data_dataframe = hasattr(data, 'columns')

    # Determine whether layout is an array of node positions
    is_layout_ndarray = hasattr(layout, 'dtype')
    if is_layout_ndarray:
        node_pos = layout
    else:
        node_pos = graph.layout(layout, dim=layout_dim)

    color_variable_kind = _infer_color_variable_kind(color_variable, data)

    # Determine whether node_colors is an array of node colors
    is_node_colors_ndarray = hasattr(node_color_statistic, 'dtype')
    if (not is_node_colors_ndarray) and (not callable(node_color_statistic)):
        raise ValueError("node_color_statistic must be a callable or ndarray.")

    node_colors = _get_node_colors(
        data, is_data_dataframe, node_elements,
        is_node_colors_ndarray, node_color_statistic,
        color_variable, color_variable_kind)

    plot_options = {
        'edge_trace_mode': 'lines',
        'edge_trace_line': dict(color='#888', width=1),
        'edge_trace_hoverinfo': 'none',
        'node_trace_mode': 'markers',
        'node_trace_hoverinfo': 'text',
        'node_trace_hoverlabel': dict(
            bgcolor=list(map(lambda x: rgb2hex(get_cmap('viridis')(x)),
                             node_colors))),
        'node_trace_marker_color': list(
            map(lambda x: rgb2hex(get_cmap('viridis')(x)), node_colors)),
        'node_trace_marker_colorscale': 'viridis',
        'node_trace_marker_showscale': True,
        'node_trace_marker_reversescale': False,
        'node_trace_marker_line': dict(width=.5, color='#888'),
        'node_trace_marker_size': _get_node_size(node_elements),
        'node_trace_marker_sizemode': 'area',
        'node_trace_marker_sizeref': set_node_sizeref(node_elements),
        'node_trace_marker_sizemin': 4,
        'node_trace_marker_cmin': np.min(node_colors),
        'node_trace_marker_cmax': np.max(node_colors),
        'node_trace_marker_colorbar': dict(thickness=15,
                                           title='',
                                           xanchor='left',
                                           titleside='right'),
        'node_trace_marker_line_width': 2,
        'node_trace_text': _get_node_text(graph),
        'layout_showlegend': False,
        'layout_hovermode': 'closest',
        'layout_xaxis_title': "",
        'layout_yaxis_title': "",
        'layout_title': "",
        'layout_margin': {'b': 20, 'l': 5, 'r': 5, 't': 40},
        'layout_annotations': list()
    }

    if plotly_kwargs is not None:
        plot_options.update(plotly_kwargs)

    edge_x = list(reduce(operator.iconcat,
                         map(lambda x: [node_pos[x[0]][0],
                                        node_pos[x[1]][0], None],
                             graph.get_edgelist()), []))
    edge_y = list(reduce(operator.iconcat,
                         map(lambda x: [node_pos[x[0]][1],
                                        node_pos[x[1]][1], None],
                             graph.get_edgelist()), []))

    node_x = [node_pos[k][0] for k in range(graph.vcount())]
    node_y = [node_pos[k][1] for k in range(graph.vcount())]

    if layout_dim == 2:
        plot_options.update({
            'layout_xaxis': dict(showgrid=False, zeroline=False,
                                 showticklabels=False, ticks="",
                                 showline=False),
            'layout_yaxis': dict(showgrid=False, zeroline=False,
                                 showticklabels=False, ticks="",
                                 showline=False),
        })
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=plot_options['edge_trace_line'],
            hoverinfo=plot_options['edge_trace_hoverinfo'],
            mode=plot_options['edge_trace_mode'])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode=plot_options['node_trace_mode'],
            hoverinfo=plot_options['node_trace_hoverinfo'],
            hovertext=plot_options['node_trace_text'],
            marker=dict(
                showscale=plot_options['node_trace_marker_showscale'],
                colorscale=plot_options['node_trace_marker_colorscale'],
                reversescale=plot_options['node_trace_marker_reversescale'],
                line=plot_options['node_trace_marker_line'],
                color=list(
                    map(lambda x: rgb2hex(
                        get_cmap(
                            plot_options['node_trace_marker_colorscale']
                        )(x)), node_colors)),
                size=plot_options['node_trace_marker_size'],
                sizemode=plot_options['node_trace_marker_sizemode'],
                sizeref=plot_options['node_trace_marker_sizeref'],
                sizemin=plot_options['node_trace_marker_sizemin'],
                cmin=plot_options['node_trace_marker_cmin'],
                cmax=plot_options['node_trace_marker_cmax'],
                colorbar=plot_options['node_trace_marker_colorbar'],
                line_width=plot_options['node_trace_marker_line_width']),
            text=plot_options['node_trace_text'])
    elif layout_dim == 3:
        plot_options.update({
            'axis': dict(showbackground=False,
                            showline=False,
                            zeroline=False,
                            showgrid=False,
                            showticklabels=False,
                            title='')
        })
        plot_options['layout_scene'] = dict(xaxis=dict(plot_options['axis']),
                                            yaxis=dict(
                                                plot_options['axis']),
                                            zaxis=dict(
                                                plot_options['axis'])
                                            )

        edge_z = list(reduce(operator.iconcat,
                             map(lambda x: [node_pos[x[0]][2],
                                            node_pos[x[1]][2], None],
                                 graph.get_edgelist()), []))

        node_z = [node_pos[k][2] for k in range(graph.vcount())]

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode=plot_options['edge_trace_mode'],
            line=plot_options['edge_trace_line'],
            hoverinfo=plot_options['edge_trace_hoverinfo'])

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode=plot_options['node_trace_mode'],
            hoverinfo=plot_options['node_trace_hoverinfo'],
            hoverlabel=plot_options['node_trace_hoverlabel'],
            hovertext=plot_options['node_trace_text'],
            marker=dict(
                showscale=plot_options['node_trace_marker_showscale'],
                colorscale=plot_options['node_trace_marker_colorscale'],
                reversescale=plot_options['node_trace_marker_reversescale'],
                line=plot_options['node_trace_marker_line'],
                color=plot_options['node_trace_marker_color'],
                size=plot_options['node_trace_marker_size'],
                sizemode=plot_options['node_trace_marker_sizemode'],
                sizeref=plot_options['node_trace_marker_sizeref'],
                sizemin=plot_options['node_trace_marker_sizemin'],
                cmin=plot_options['node_trace_marker_cmin'],
                cmax=plot_options['node_trace_marker_cmax'],
                colorbar=plot_options['node_trace_marker_colorbar'],
                line_width=plot_options['node_trace_marker_line_width']),
            text=plot_options['node_trace_text'])

    return node_trace, edge_trace, node_elements, node_colors, plot_options
