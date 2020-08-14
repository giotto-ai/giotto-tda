"""Testing for interactivity in Mapper plotting functions."""
# License: GNU AGPLv3

import warnings

import numpy as np

from gtda.mapper import FirstSimpleGap, make_mapper_pipeline, \
    plot_interactive_mapper_graph

N = 50
d = 3
X = np.random.randn(N, d)


def _get_widget_by_trait(fig, key, val=None):
    for k, v in fig.widgets.items():
        try:
            b = getattr(v, key) == val if val is not None \
                else getattr(v, key)
            if b:
                return fig.widgets[k]
        except (AttributeError, TypeError):
            pass


def test_pipeline_change_not_cloned():
    """Verify that the pipeline is changed on interaction if `clone_pipeline`
    is True."""
    warnings.simplefilter("ignore")
    initial_affin = 'euclidean'
    new_affin = 'manhattan'

    pipe = make_mapper_pipeline(
        clusterer=FirstSimpleGap(affinity=initial_affin)
        )
    fig = plot_interactive_mapper_graph(pipe, X, clone_pipeline=False)

    # Get widget and change the affinity type
    w_text = _get_widget_by_trait(fig, 'description', 'affinity')
    w_text.set_state({'value': new_affin})
    final_affin = pipe.get_mapper_params()['clusterer__affinity']
    assert final_affin == new_affin
