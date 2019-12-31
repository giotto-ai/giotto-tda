import numpy as np

"""Graph layout functions"""


def _get_node_size(node_elements):
    # TODO: add doc strings to all functions
    return list(map(len, node_elements))


def _get_node_text(graph):
    return [
        'Node Id:{}<br>Node size:{}<br>Pullback Id:{}<br>Cluster label:{}'
        .format(
            node_id, len(node_elements), interval_id, cluster_id,
        )
        for node_id, node_elements, interval_id, cluster_id in
        zip(graph['node_metadata']['node_id'],
            graph['node_metadata']['node_elements'],
            graph['node_metadata']['interval_id'],
            graph['node_metadata']['cluster_id'])]


def set_node_sizeref(node_elements, node_scale=12):
    # Formula from Plotly https://plot.ly/python/bubble-charts/
    return 2. * max(_get_node_size(node_elements)) / (node_scale ** 2)


def get_node_summary(node_elements, data, summary_stat=np.mean):
    return list(map(summary_stat, [data[itr] for itr in node_elements]))


"""Plotly layout functions"""


def _get_colorscales():
    # TODO: move this to utils
    return ['Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 'Greens',
            'Greys', 'Hot', 'Jet', 'Picnic', 'Portland', 'Rainbow', 'RdBu',
            'Reds', 'Viridis', 'YlGnBu', 'YlOrRd']


def _get_column_color_buttons(data, is_data_dataframe, node_elements,
                              node_colors_color_variable):
    # TODO: Consider opting for on-demand computation instead of precomputing
    #  all node summary values when this is called by viz functions
    if is_data_dataframe:
        columns_to_color = data.columns
    else:
        columns_to_color = range(data.shape[1])

    column_color_buttons = [
        dict(
            args=[{
                'marker.color': [None, node_colors_color_variable],
                'marker.cmin': [None, np.min(node_colors_color_variable)],
                'marker.cmax': [None, np.max(node_colors_color_variable)]
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

        column_color_buttons.append(
            dict(
                args=[{
                    'marker.color': [None, node_colors],
                    'marker.cmin': [None, np.min(node_colors)],
                    'marker.cmax': [None, np.max(node_colors)]
                }],
                label='Column {}'.format(column),
                method='restyle'
            )
        )
    return column_color_buttons


def _get_colorscale_buttons(colorscales):
    colorscale_buttons = []
    for colorscale in colorscales:
        colorscale_buttons.append(
            dict(
                args=[{'marker.colorscale': [None, colorscale]}],
                label=colorscale,
                method='restyle'
            )
        )
    return colorscale_buttons
