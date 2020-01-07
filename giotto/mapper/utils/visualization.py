import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

"""Graph layout functions"""


def _get_node_size(node_elements):
    # TODO: add doc strings to all functions
    return list(map(len, node_elements))


def _get_node_text(graph):
    return [
        'Node Id:{}<br>Node size:{}<br>Cluster label:{}'
        .format(
            interval_id, len(node_elements), cluster_id,
        )
        for interval_id, node_elements, cluster_id in
        zip(graph['node_metadata']['interval_id'],
            graph['node_metadata']['node_elements'],
            graph['node_metadata']['cluster_id'])]


def set_node_sizeref(node_elements, node_scale=12):
    # Formula from Plotly https://plot.ly/python/bubble-charts/
    return 2. * max(_get_node_size(node_elements)) / (node_scale ** 2)


def get_node_summary(node_elements, data, summary_stat=np.mean):
    return np.asarray(
        list(map(summary_stat, [data[itr] for itr in node_elements]))
    )


"""Plotly layout functions"""


def _get_column_color_buttons(data, is_data_dataframe, node_elements,
                              node_colors_color_variable, colorscale):
    # TODO: Consider opting for on-demand computation instead of precomputing
    #  all node summary values when this is called by viz functions
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
                label='Column {}'.format(column),
                method='restyle'
            )
        )
    return column_color_buttons


def _is_array_or_dataframe(color_variable, data):
    """Determines whether color_variable is array or pandas dataframe."""
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
        color_variable_kind = 'data'
    else:  # Assume color_variable is a selection of columns
        color_variable_kind = 'columns'

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
        elif color_variable_kind == 'data':
            if is_data_dataframe:
                color_data = data.to_numpy()
            else:
                color_data = data
        else:
            if is_data_dataframe:
                color_data = data[color_variable].to_numpy()
            else:
                color_data = data[:, color_variable]
        node_colors = get_node_summary(node_elements, color_data,
                                       summary_stat=node_color_statistic)

    return node_colors
