"""Testing for interactivity in Mapper plotting functions."""
# License: GNU AGPLv3

import numpy as np
import pytest
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


@pytest.mark.parametrize('clone_pipeline', [True, False])
def test_pipeline_cloned(clone_pipeline):
    """Verify that the pipeline is not changed on interaction if and only if
    `clone_pipeline` is True."""
    initial_affin = 'euclidean'
    new_affin = 'manhattan'

    pipe = make_mapper_pipeline(
        clusterer=FirstSimpleGap(affinity=initial_affin)
        )
    fig = plot_interactive_mapper_graph(pipe, X, clone_pipeline=clone_pipeline)

    # Get widget and change the affinity type
    w_text = _get_widget_by_trait(fig, 'description', 'affinity')
    w_text.set_state({'value': new_affin})
    final_affin = pipe.get_mapper_params()['clusterer__affinity']
    assert final_affin == initial_affin if clone_pipeline else new_affin
