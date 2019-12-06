import plotly.graph_objects as go
from functools import reduce
import operator


def create_network_2d(graph, pos, node_size, node_color, node_text,
                      node_scale=12, colorscale='viridis', cmin=0, cmax=1,
                      legend_title=''):

    edge_x = list(reduce(operator.iconcat,
                         map(lambda x: [pos[x[0]][0],
                                        pos[x[1]][0], None],
                             graph.edges()), []))
    edge_y = list(reduce(operator.iconcat,
                         map(lambda x: [pos[x[0]][1],
                                        pos[x[1]][1], None],
                             graph.edges()), []))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[k][0] for k in graph.nodes()]
    node_y = [pos[k][1] for k in graph.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            reversescale=True,
            line=dict(width=.5, color='#888'),
            color=node_color,
            size=node_size,
            sizemode='area',
            sizeref=2.*max(node_size)/(node_scale**2),
            sizemin=4,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                thickness=15,
                title=legend_title,
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        text=node_text)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False,
                                     hovermode='closest',
                                     margin={'b': 20, 'l': 5, 'r': 5, 't': 40},
                                     xaxis=dict(showgrid=False, zeroline=False,
                                                showticklabels=False, ticks="",
                                                showline=False),
                                     yaxis=dict(showgrid=False, zeroline=False,
                                                showticklabels=False, ticks="",
                                                showline=False),
                                     xaxis_title="",
                                     yaxis_title=""))
    fig.update_layout(template='simple_white')

    return fig


def create_network_3d(graph, pos, size, node_color, node_text, node_scale=12,
                      colorscale='viridis', cmin=0, cmax=1,
                      legend_title=''):

    edge_x = list(reduce(operator.iconcat,
                         map(lambda x: [pos[x[0]][0],
                                        pos[x[1]][0], None],
                             graph.edges()), []))
    edge_y = list(reduce(operator.iconcat,
                         map(lambda x: [pos[x[0]][1],
                                        pos[x[1]][1], None],
                             graph.edges()), []))

    edge_z = list(reduce(operator.iconcat,
                         map(lambda x: [pos[x[0]][2],
                                        pos[x[1]][2], None],
                             graph.edges()), []))

    edge_trace = go.Scatter3d(x=edge_x,
                              y=edge_y,
                              z=edge_z,
                              mode='lines',
                              line=dict(color='rgb(125,125,125)',
                                        width=1),
                              hoverinfo='none')

    node_x = [pos[k][0] for k in graph.nodes()]
    node_y = [pos[k][1] for k in graph.nodes()]
    node_z = [pos[k][2] for k in graph.nodes()]

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            reversescale=True,
            line=dict(width=.5, color='#888'),
            color=node_color,
            size=size,
            sizemode='area',
            sizeref=2.*max(size)/(node_scale**2),
            sizemin=4,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                thickness=15,
                title=legend_title,
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        text=node_text,
        hoverinfo='text')

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    layout = go.Layout(
        title="",
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(xaxis=dict(axis),
                   yaxis=dict(axis),
                   zaxis=dict(axis)),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[])

    data = [edge_trace, node_trace]
    fig = go.Figure(data=data, layout=layout)

    return fig
