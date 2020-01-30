import plotly.io as pio
import numpy as np
import pandas as pd
import warnings

from unittest import TestCase
from numpy.testing import assert_almost_equal, assert_raises

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
colors = np.array([8.,  8.,  3.,  8.,  0.,  8.,  8.,  8.,  5.,  8.,  8.,  8.,  8.,
                   4.,  2., 8., 1.,  8., 2.,  8.])


class TestStaticPlot(TestCaseNoTemplate):

    def test_is_data_present(self):
        pipe = make_mapper_pipeline()
        warnings.simplefilter("ignore")
        fig = plot_static_mapper_graph(pipe, X,
                                       color_variable=colors,
                                       clone_pipeline=False)
        xy = np.stack([fig.get_state()['_data'][1][c] for c in ['x', 'y']]).transpose()
        assert X.shape >= xy.shape

        real_colors = fig.get_state()['_data'][1]['marker']['color']
        assert len(real_colors) == xy.shape[0]


class TestInteractivePlot(TestCaseNoTemplate):

    def test_kind_changes(self):
        pipe = make_mapper_pipeline(clusterer=FirstSimpleGap())
        warnings.simplefilter("ignore")
        fig = plot_interactive_mapper_graph(pipe, X)

        def get_widget_by_trait(key, val=None):
            for k, v in fig.widgets.items():
                try:
                    b = getattr(v, key) == val if val is not None\
                        else getattr(v, key)
                    if b:
                        return fig.widgets[k]
                except (AttributeError, TypeError):
                    pass
        w_scatter = get_widget_by_trait('data', val=None)
        old_pts = np.array([w_scatter.data[1][c] for c in ['x', 'y']])

        w = get_widget_by_trait('description', 'kind')
        w.set_trait('value', 'balanced')
        w_scatter_new = get_widget_by_trait('data', val=None)
        new_pts = np.array([w_scatter_new.data[1][c] for c in ['x', 'y']])

        try:
            assert_raises(AssertionError, assert_almost_equal, old_pts, new_pts)
        except AssertionError as e:
            print(e)
            print(old_pts, new_pts)
            print(w)
            raise AssertionError(e)
