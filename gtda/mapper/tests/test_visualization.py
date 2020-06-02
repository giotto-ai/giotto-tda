import plotly.io as pio
import numpy as np
import warnings

from unittest import TestCase

from gtda.mapper import make_mapper_pipeline
from gtda.mapper import (plot_interactive_mapper_graph,
                         plot_static_mapper_graph)
from gtda.mapper import FirstSimpleGap


class TestCaseNoTemplate(TestCase):
    def setUp(self):
        pio.templates.default = None

    def tearDown(self):
        pio.templates.default = "plotly"


X = np.array([[-19.33965799, -284.58638371],
              [-290.25710696,  184.31095197],
              [250.38108853,  134.5112574],
              [-259.46357187, -172.12937543],
              [115.72180479,  -69.67624071],
              [120.12187185,  248.39783826],
              [234.08476944,  115.54743986],
              [246.68634685,  119.170029],
              [-154.27214561, -272.07656956],
              [225.37435664,  186.3253872],
              [54.17543392,   76.4066916],
              [175.28163213, -193.46279193],
              [228.63910018, -121.16687597],
              [-101.58902866,   48.86471748],
              [-185.23421146,  244.14414753],
              [-275.05799067, -204.99265911],
              [-170.12180583,  176.10258455],
              [-155.54055842, -214.420498],
              [184.6940872,    2.08810678],
              [-184.42012962,   28.8978038]])
colors = np.array([8., 8., 3., 8., 0., 8., 8., 8., 5.,
                   8., 8., 8., 8., 4., 2., 8., 1., 8., 2., 8.])


class TestStaticPlot(TestCaseNoTemplate):

    def test_is_data_present(self):
        """Verify that what we see in the graph corresponds to
        the number of samples in the graph."""
        pipe = make_mapper_pipeline()
        warnings.simplefilter("ignore")
        fig = plot_static_mapper_graph(pipe, X,
                                       color_variable=colors,
                                       clone_pipeline=False)
        node_trace_x = fig.get_state()['_data'][1]["x"]
        node_trace_y = fig.get_state()['_data'][1]["y"]

        assert node_trace_x["shape"][0] == node_trace_y["shape"][0]

        num_nodes = node_trace_x["shape"][0]
        assert len(X) >= num_nodes

        fig_colors = fig.get_state()['_data'][1]['marker']['color']
        assert len(fig_colors) == num_nodes


class TestInteractivePlot(TestCaseNoTemplate):

    def _get_widget_by_trait(self, fig, key, val=None):
        for k, v in fig.widgets.items():
            try:
                b = getattr(v, key) == val if val is not None \
                    else getattr(v, key)
                if b:
                    return fig.widgets[k]
            except (AttributeError, TypeError):
                pass

    def _get_size_from_hovertext(self, s):
        size_str = s.split("<br>")[1].split(": ")[1]
        return int(size_str)

    def test_cluster_sizes(self):
        """Verify that the total number of calculated clusters is equal to
        the number of displayed clusters."""
        pipe = make_mapper_pipeline(clusterer=FirstSimpleGap())
        warnings.simplefilter("ignore")
        fig = plot_interactive_mapper_graph(pipe, X)
        w_scatter = self._get_widget_by_trait(fig, 'data')

        node_sizes_vis = [self._get_size_from_hovertext(s_)
                          for s_ in w_scatter.get_state()
                          ['_data'][1]['hovertext']]

        g = pipe.fit_transform(X)
        node_size_real = [len(node)
                          for node in g['node_metadata']['node_elements']]

        assert sum(node_sizes_vis) == sum(node_size_real)
