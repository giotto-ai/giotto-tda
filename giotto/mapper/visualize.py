import logging
import operator
import traceback
from functools import reduce

import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import Layout, widgets
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex


class OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget"""

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': '100%',
            'height': '160px',
            'border': '1px solid black',
            'overflow_y': 'auto'
        }
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """Overload of logging.Handler method"""
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record+'\n'
        }
        self.out.outputs = (new_output, ) + self.out.outputs

    def show_logs(self):
        """Show the logs"""
        display(self.out)

    def clear_logs(self):
        """Clear the current logs"""
        self.out.clear_output()


def get_node_size(node_elements):
    # TODO: add doc strings to all functions
    return list(map(len, node_elements))


def get_node_text(graph):
    return graph.vs.get_attribute_values('name')


def create_network_2d(graph, node_pos, node_color, node_scale=12,
                      custom_plot_options=None):
    # TODO: allow custom size reference
    node_elements = graph.vs.get_attribute_values('elements')
    plot_options = {
        'edge_trace_line': dict(width=0.5, color='#888'),
        'edge_trace_hoverinfo': 'none',
        'edge_trace_mode': 'lines',
        'node_trace_mode': 'markers',
        'node_trace_hoverinfo': 'text',
        'node_trace_marker_showscale': True,
        'node_trace_marker_colorscale': 'viridis',
        'node_trace_marker_reversescale': False,
        'node_trace_marker_line': dict(width=.5, color='#888'),
        'node_trace_marker_color': node_color,
        'node_trace_marker_size': get_node_size(node_elements),
        'node_trace_marker_sizemode': 'area',
        'node_trace_marker_sizeref':
            2. * max(get_node_size(node_elements)) / (node_scale ** 2),
        'node_trace_marker_sizemin': 4,
        'node_trace_marker_cmin': 0,
        'node_trace_marker_cmax': 1,
        'node_trace_marker_colorbar': dict(thickness=15,
                                           title='',
                                           xanchor='left',
                                           titleside='right'),
        'node_trace_marker_line_width': 2,
        'node_trace_text': get_node_text(graph),
        'layout_showlegend': False,
        'layout_hovermode': 'closest',
        'layout_margin': {'b': 20, 'l': 5, 'r': 5, 't': 40},
        'layout_xaxis': dict(showgrid=False, zeroline=False,
                             showticklabels=False, ticks="",
                             showline=False),
        'layout_yaxis': dict(showgrid=False, zeroline=False,
                             showticklabels=False, ticks="",
                             showline=False),
        'layout_xaxis_title': "",
        'layout_yaxis_title': ""
    }

    if custom_plot_options is None:
        plot_options.update({})
    else:
        plot_options.update(custom_plot_options)

    # TODO check we are not losing performance by using map + lambda
    edge_x = list(reduce(operator.iconcat,
                         map(lambda x: [node_pos[x[0]][0],
                                        node_pos[x[1]][0], None],
                             graph.get_edgelist()), []))

    edge_y = list(reduce(operator.iconcat,
                         map(lambda x: [node_pos[x[0]][1],
                                        node_pos[x[1]][1], None],
                             graph.get_edgelist()), []))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=plot_options['edge_trace_line'],
        hoverinfo=plot_options['edge_trace_hoverinfo'],
        mode=plot_options['edge_trace_mode'])

    node_x = [node_pos[k][0] for k in range(graph.vcount())]
    node_y = [node_pos[k][1] for k in range(graph.vcount())]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode=plot_options['node_trace_mode'],
        hoverinfo=plot_options['node_trace_hoverinfo'],
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

    fig = go.FigureWidget(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=plot_options['layout_showlegend'],
            hovermode=plot_options['layout_hovermode'],
            margin=plot_options['layout_margin'],
            xaxis=plot_options['layout_xaxis'],
            yaxis=plot_options['layout_yaxis'],
            xaxis_title=plot_options['layout_xaxis_title'],
            yaxis_title=plot_options['layout_yaxis_title'])
    )
    fig.update_layout(template='simple_white')

    # Add dropdown for colorscale of nodes
    # TODO consider fishing available colorscales from plotly and generating
    #  all possibilities via a list comprehension
    button_height = 1.
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=list([
                    dict(
                        args=[
                            {"marker.colorscale": [None, 'Viridis']}],
                        label="Viridis",
                        method="restyle"
                    ),
                    dict(
                        args=[{"marker.colorscale": [None, "Cividis"]}],
                        label="Cividis",
                        method="restyle"
                    ),
                    dict(
                        args=[{"marker.colorscale": [None, "Blues"]}],
                        label="Blues",
                        method="restyle"
                    ),
                    dict(
                        args=[{"marker.colorscale": [None, "Greens"]}],
                        label="Greens",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                y=button_height,
                yanchor="top"
            )])

    return fig


def create_network_3d(graph, node_pos, node_color, node_scale=12,
                      custom_plot_options=None):
    node_elements = graph.vs.get_attribute_values('elements')
    plot_options = {
        'edge_trace_mode': 'lines',
        'edge_trace_line': dict(color='rgb(125,125,125)',
                                width=1),
        'edge_trace_hoverinfo': 'none',
        'node_trace_mode': 'markers',
        'node_trace_hoverinfo': 'text',
        'node_trace_hoverlabel': dict(
            bgcolor=list(map(lambda x: rgb2hex(get_cmap('viridis')(x)),
                             node_color))),
        'node_trace_marker_showscale': True,
        'node_trace_marker_colorscale': 'viridis',
        'node_trace_marker_reversescale': False,
        'node_trace_marker_line': dict(width=.5, color='#888'),
        'node_trace_marker_color': node_color,
        'node_trace_marker_size': get_node_size(node_elements),
        'node_trace_marker_sizemode': 'area',
        'node_trace_marker_sizeref':
            2. * max(get_node_size(node_elements)) / (node_scale ** 2),
        'node_trace_marker_sizemin': 4,
        'node_trace_marker_cmin': 0,
        'node_trace_marker_cmax': 1,
        'node_trace_marker_colorbar': dict(thickness=15,
                                           title='',
                                           xanchor='left',
                                           titleside='right'),
        'node_trace_marker_line_width': 2,
        'node_trace_text': get_node_text(graph),
        'axis': dict(showbackground=False,
                     showline=False,
                     zeroline=False,
                     showgrid=False,
                     showticklabels=False,
                     title=''),
        'layout_title': "",
        'layout_width': 1000,
        'layout_height': 1000,
        'layout_showlegend': False,
        'layout_margin': dict(t=100),
        'layout_hovermode': 'closest',
        'layout_annotations': []
    }

    plot_options['layout_scene'] = dict(xaxis=dict(plot_options['axis']),
                                        yaxis=dict(plot_options['axis']),
                                        zaxis=dict(plot_options['axis']))

    if custom_plot_options is None:
        plot_options.update({})
    else:
        plot_options.update(custom_plot_options)

    edge_x = list(reduce(operator.iconcat,
                         map(lambda x: [node_pos[x[0]][0],
                                        node_pos[x[1]][0], None],
                             graph.get_edgelist()), []))
    edge_y = list(reduce(operator.iconcat,
                         map(lambda x: [node_pos[x[0]][1],
                                        node_pos[x[1]][1], None],
                             graph.get_edgelist()), []))

    edge_z = list(reduce(operator.iconcat,
                         map(lambda x: [node_pos[x[0]][2],
                                        node_pos[x[1]][2], None],
                             graph.get_edgelist()), []))

    edge_trace = go.Scatter3d(x=edge_x,
                              y=edge_y,
                              z=edge_z,
                              mode=plot_options['edge_trace_mode'],
                              line=plot_options['edge_trace_line'],
                              hoverinfo=plot_options['edge_trace_hoverinfo'])

    node_x = [node_pos[k][0] for k in range(graph.vcount())]
    node_y = [node_pos[k][1] for k in range(graph.vcount())]
    node_z = [node_pos[k][2] for k in range(graph.vcount())]

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode=plot_options['node_trace_mode'],
        hoverinfo=plot_options['node_trace_hoverinfo'],
        hoverlabel=plot_options['node_trace_hoverlabel'],
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

    layout = go.Layout(
        title=plot_options['layout_title'],
        width=plot_options['layout_width'],
        height=plot_options['layout_height'],
        showlegend=plot_options['layout_showlegend'],
        scene=plot_options['layout_scene'],
        margin=plot_options['layout_margin'],
        hovermode=plot_options['layout_hovermode'],
        annotations=plot_options['layout_annotations'])

    data = [edge_trace, node_trace]
    fig = go.Figure(data=data, layout=layout)

    return fig


def create_interactive_network(pipe, data, plotly_kwargs=None, node_pos=None,
                               node_color=None, summary_stat=np.mean, dim=2):
    # TODO could abstract away common patterns in get_cover_params_widgets and
    #  get_cluster_params_widgets

    # TODO allow dimension to be passed as either 2 or 3 as an arg or kwarg

    def get_node_summary(node_elements, data, summary_stat=np.mean):
        return list(map(lambda x: summary_stat(data[x]),
                        node_elements))

    def get_cover_params_widgets(param, value):
        if isinstance(value, float):
            return (param, widgets.FloatSlider(
                value=value,
                step=0.1,
                min=0.1,
                max=1.0,
                description=param.split('__')[1],
                disabled=False
            ))
        elif isinstance(value, int):
            return (param, widgets.IntSlider(
                value=value,
                min=1,
                max=50,
                step=1,
                description=param.split('__')[1],
                disabled=False
            ))
        else:
            return None

    def get_cluster_params_widgets(param, value):
        if isinstance(value, float):
            return (param, widgets.FloatText(
                    value=value,
                    step=0.1,
                    description=param.split('__')[1],
                    disabled=False
                    ))
        elif isinstance(value, int):
            return (param, widgets.IntText(
                value=value,
                step=1,
                description=param.split('__')[1],
                disabled=False
            ))
        elif isinstance(value, str):
            return (param, widgets.Text(
                value=value,
                description=param.split('__')[1],
                disabled=False
            ))
        else:
            return None

    def update_figure(old_figure, new_figure):
        # TODO could this be abstracted to node and edge traces and metadata
        #  information without need for creating a full new figure object
        old_figure.data[0].x = new_figure.data[0].x
        old_figure.data[0].y = new_figure.data[0].y
        old_figure.data[1].x = new_figure.data[1].x
        old_figure.data[1].y = new_figure.data[1].y
        old_figure.data[1].marker.size = new_figure.data[1].marker.size
        old_figure.data[1].marker.color = new_figure.data[1].marker.color
        old_figure.data[1].marker.sizeref = new_figure.data[1].marker.sizeref

    def get_figure(pipe, data, node_pos, node_color, summary_stat):
        graph = pipe.fit_transform(data)
        node_elements = graph.vs.get_attribute_values('elements')
        if node_pos is None:
            node_pos = graph.layout('kamada_kawai')

        if node_color is None:
            node_color = get_node_summary(node_elements, data,
                                          summary_stat=summary_stat)

        return create_network_2d(graph, node_pos, node_color, **plotly_kwargs)

    def response_numeric(change):
        # TODO: remove hardcoding of keys and mimic what is done with clusterer
        handler.clear_logs()
        try:
            pipe.set_mapper_params(
                cover__n_intervals=cover_params_widgets['cover__n_intervals']
                .value)
            pipe.set_mapper_params(
                cover__overlap_frac=cover_params_widgets['cover__overlap_frac']
                .value)

            for param, value in cluster_params.items():
                if isinstance(value, (int, float)):
                    pipe.set_mapper_params(
                        **{param: cluster_params_widgets[param].value}
                    )

            # TODO check this alternative:
            #
            # num_params = {param: value for param, value in cluster_params.items()
            #               if isinstance(value, (int, float))}
            #
            # pipe.set_mapper_params(
            #     **{param: cluster_params_widgets[param].value for param in
            #        num_params}
            # )

            new_fig = get_figure(pipe, data, node_pos,
                                 node_color, summary_stat)

            logger.info("Updating figure ...")
            with fig.batch_update():
                update_figure(fig, new_fig)
            valid.value = True
        except Exception:
            exception_data = traceback.format_exc().splitlines()
            logger.exception(exception_data[-1])
            valid.value = False

    def response_text(text):
        handler.clear_logs()
        try:
            for param, value in cluster_params.items():
                if isinstance(value, str):
                    pipe.set_mapper_params(
                        **{param: cluster_params_widgets[param].value}
                    )

            new_fig = get_figure(pipe, data, node_pos,
                                 node_color, summary_stat)

            logger.info("Updating figure ...")
            with fig.batch_update():
                update_figure(fig, new_fig)
            valid.value = True
        except Exception:
            exception_data = traceback.format_exc().splitlines()
            logger.exception(exception_data[-1])
            valid.value = False

    def observe_numeric_widgets(params, widgets):
        for param, value in params.items():
            if isinstance(value, (int, float)):
                widgets[param].observe(response_numeric, names='value')

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
        '%(asctime)s  - [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # initialise cover and cluster dictionaries of parameters and widgets
    cover_params = dict(filter(lambda x: x[0].startswith('cover'),
                               pipe.get_mapper_params().items()))
    cover_params_widgets = dict(
        filter(None, map(lambda x: get_cover_params_widgets(*x),
                         cover_params.items())))
    cluster_params = dict(filter(lambda x: x[0].startswith('clusterer'),
                                 pipe.get_mapper_params().items()))
    cluster_params_widgets = dict(
        filter(None, map(lambda x: get_cluster_params_widgets(*x),
                         cluster_params.items())))

    # create button for text inputs
    submit_button = widgets.Button(description="Submit")
    submit_button.on_click(response_text)

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

    fig = get_figure(pipe, data, node_pos, node_color, summary_stat)

    observe_numeric_widgets(cover_params, cover_params_widgets)
    observe_numeric_widgets(cluster_params, cluster_params_widgets)

    logs_box.observe(click_box, names='value')

    # define containers for input widgets
    container_cover = widgets.HBox(children=list(
        cover_params_widgets.values()))

    container_cluster_text = widgets.HBox(
        children=list(v for k, v in cluster_params_widgets.items()
                      if isinstance(cluster_params[k], str)) + [submit_button])

    container_cluster_layout = Layout(display='flex', flex_flow='row wrap')

    container_cluster_numeric = widgets.HBox(
        children=list(v for k, v in cluster_params_widgets.items()
                      if isinstance(cluster_params[k], (int, float))
                      ), layout=container_cluster_layout
    )

    box = widgets.VBox([container_cover, container_cluster_text,
                        container_cluster_numeric, fig,
                        valid, logs_box])
    display(box, out)
