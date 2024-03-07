import math
import pytest

from gtda.externals.modules.gtda_wasserstein import wasserstein_distance, \
    hera_get_infinity


def test_trivial_empty_diagram():
    diagram_a = []
    diagram_b = []
    q = 1.0
    delta = 0.01
    internal_p = float('inf')
    initial_eps = 0.0
    eps_factor = 0.0
    max_bids_per_round = 0

    assert 0 == wasserstein_distance(diagram_a, diagram_b, q=q, delta=delta,
                                     internal_p=internal_p,
                                     initial_eps=initial_eps,
                                     eps_factor=eps_factor,
                                     max_bids_per_round=max_bids_per_round)


def test_trivial_1():
    """
    trivial: one empty diagram, one single-point diagram
    """
    diagram_a = [[1.0, 2.0]]
    diagram_b = []
    q = 1.0
    delta = 0.01
    internal_p = hera_get_infinity()
    initial_eps = 0.0
    eps_factor = 0.0
    max_bids_per_round = 1

    assert 0 >= wasserstein_distance(diagram_a, diagram_b, q=q, delta=delta,
                                     internal_p=internal_p,
                                     initial_eps=initial_eps,
                                     eps_factor=eps_factor,
                                     max_bids_per_round=max_bids_per_round) \
        - 0.5

    assert 0 >= wasserstein_distance(diagram_b, diagram_a, q=q, delta=delta,
                                     internal_p=internal_p,
                                     initial_eps=initial_eps,
                                     eps_factor=eps_factor,
                                     max_bids_per_round=max_bids_per_round) \
        - 0.5

    internal_p = 2.0
    correct_answer = 1.0 / math.sqrt(2.0)

    assert 0 == pytest.approx(
        wasserstein_distance(diagram_b, diagram_a, q=q, delta=delta,
                             internal_p=internal_p, initial_eps=initial_eps,
                             eps_factor=eps_factor,
                             max_bids_per_round=max_bids_per_round)
        - correct_answer)


def test_trivial_2():
    """
    trivial: two single-point diagrams-1
    """
    diagram_a = [[10.0, 20.0]]
    diagram_b = [[13.0, 19.0]]
    q = 1.0
    delta = 0.01
    internal_p = hera_get_infinity()
    initial_eps = 0.0
    eps_factor = 0.0
    max_bids_per_round = 0

    d1 = wasserstein_distance(diagram_a, diagram_b, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)
    d2 = wasserstein_distance(diagram_b, diagram_a, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)

    assert d1 == d2
    assert d1 == 3.0


def test_inf_1():
    """
    two points at infinity, no finite points
    """
    # edge cost 1.0
    diagram_a = [[1, float('inf')]]
    diagram_b = [[2, float('inf')]]

    q = 1.0
    delta = 0.01
    internal_p = hera_get_infinity()
    initial_eps = 0.0
    eps_factor = 0.0
    max_bids_per_round = 0
    correct_answer = 1.0

    d1 = wasserstein_distance(diagram_a, diagram_b, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)

    assert correct_answer == d1


def test_inf_2():
    """
    two points at infinity
    """
    # edge cost 3.0
    diagram_a = [[10.0, 20.0]]
    diagram_b = [[13.0, 19.0]]
    # edge cost 1.0
    diagram_a.append([1, float('inf')])
    diagram_b.append([2, float('inf')])

    q = 1.0
    delta = 0.01
    internal_p = hera_get_infinity()
    initial_eps = 0.0
    eps_factor = 0.0
    max_bids_per_round = 0
    correct_answer = 3.0 + 1.0

    d1 = wasserstein_distance(diagram_a, diagram_b, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)

    assert correct_answer == d1


def test_inf_3():
    """
    simple small example with finite answer
    """
    diagram_a = [[1, float('inf')]]
    diagram_b = [[2, float('inf')]]

    diagram_a.append([1.9, float('inf')])
    diagram_b.append([1.1, float('inf')])

    # 1.1 - 1.0 +  2.0 - 1.9 = 0.2

    diagram_a.append([float('inf'), 1])
    diagram_b.append([float('inf'), 2])

    diagram_a.append([float('inf'), 1.9])
    diagram_b.append([float('inf'), 1.1])

    # finite edge
    diagram_a.append([10.0, 20.0])
    diagram_b.append([13.0, 19.0])

    q = 1.0
    delta = 0.01
    internal_p = hera_get_infinity()
    initial_eps = 0.0
    eps_factor = 0.0
    max_bids_per_round = 0
    correct_answer = 3.0 + 0.2 + 0.2

    d1 = wasserstein_distance(diagram_a, diagram_b, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)
    d2 = wasserstein_distance(diagram_b, diagram_a, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)
    assert d1 == d2
    assert d1 == pytest.approx(correct_answer)

    q = 2.0
    d1 = wasserstein_distance(diagram_a, diagram_b, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)
    d2 = wasserstein_distance(diagram_b, diagram_a, q=q, delta=delta,
                              internal_p=internal_p, initial_eps=initial_eps,
                              eps_factor=eps_factor,
                              max_bids_per_round=max_bids_per_round)
    corr_answer = math.sqrt(3.0 * 3.0 + 4 * 0.1 * 0.1)

    assert d1 == d2
    assert d1 == pytest.approx(corr_answer)
