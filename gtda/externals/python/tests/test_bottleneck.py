from ...modules.gtda_bottleneck import bottleneck_distance


def test_trivial_empty_diagram():
    delta = 0.01
    diagram_a = []
    diagram_b = []

    assert 0 == bottleneck_distance(diagram_a, diagram_b, delta)


def test_trivial_one_single_point():
    delta = 0.01
    diagram_a = [[1.0, 2.0]]
    diagram_b = []

    assert 0 == (bottleneck_distance(diagram_a, diagram_b, delta) - 0.5)
    assert 0 == (bottleneck_distance(diagram_b, diagram_a, delta) - 0.5)


def test_trivial_two_point_diagram_1():
    delta = 0.01
    diagram_a = [[10.0, 20.0]]
    diagram_b = [[13.0, 19.0]]
    correct_answer = 3.0

    assert (correct_answer * delta) <= (bottleneck_distance(diagram_a,
                                                            diagram_b, delta) -
                                        0.5)
    assert (correct_answer * delta) <= (bottleneck_distance(diagram_b,
                                                            diagram_a, delta) -
                                        0.5)


def test_trivial_two_point_diagram_2():
    delta = 0.01
    diagram_a = [[10.0, 20.0]]
    diagram_b = [[130.0, 138.0]]
    correct_answer = 5.0

    assert (correct_answer * delta) <= (bottleneck_distance(diagram_a,
                                                            diagram_b, delta) -
                                        0.5)
    assert (correct_answer * delta) <= (bottleneck_distance(diagram_b,
                                                            diagram_a, delta) -
                                        0.5)


def test_inf_1():
    """
    two points at infinity, no finite points
    """

    delta = 0.01
    diagram_a = [[1.0, float('inf')]]
    diagram_b = [[2.0, float('inf')]]
    correct_answer = 1.0

    assert (correct_answer * delta) <= (bottleneck_distance(diagram_a,
                                                            diagram_b, delta) -
                                        0.5)
    assert (correct_answer * delta) <= (bottleneck_distance(diagram_b,
                                                            diagram_a, delta) -
                                        0.5)


def test_inf_2():
    """
    two points at infinity
    """

    delta = 0.01
    # edge cost 1.0
    diagram_a = [[1.0, float('inf')]]
    diagram_b = [[2.0, float('inf')]]
    # edge cost 3.0
    diagram_a.append([10.0, 20.0])
    diagram_b.append([13.0, 19.0])
    correct_answer = 3.0

    assert (correct_answer * delta) <= (bottleneck_distance(diagram_a,
                                                            diagram_b, delta) -
                                        0.5)
    assert (correct_answer * delta) <= (bottleneck_distance(diagram_b,
                                                            diagram_a, delta) -
                                        0.5)


def test_inf_3():
    """
    all four corners at infinity, with finite points, infinite answer-1
    """

    delta = 0.01

    diagram_a = [[1.0, float('inf')]]
    diagram_b = [[2.0, float('inf')]]

    diagram_a.append([1, float('inf')])

    diagram_a.append([1, -float('inf')])
    diagram_b.append([2, -float('inf')])

    diagram_a.append([float('inf'), 1])
    diagram_b.append([float('inf'), 2])

    diagram_a.append([-float('inf'), 1])
    diagram_b.append([-float('inf'), 2])

    diagram_a.append([10.0, 20.0])
    diagram_b.append([13.0, 19.0])
    correct_answer = float('inf')

    assert (correct_answer) == (bottleneck_distance(diagram_a, diagram_b,
                                                    delta))
    assert (correct_answer) == (bottleneck_distance(diagram_b, diagram_a,
                                                    delta))


def test_inf_4():
    """
    simple small example with finite answer
    """

    diagram_a = [[1.0, float('inf')]]
    diagram_b = [[2.0, float('inf')]]

    diagram_a.append([1.9, float('inf')])
    diagram_b.append([1.1, float('inf')])

    # 1.1 - 1.0 +  2.0 - 1.9 = 0.2

    diagram_a.append([float('inf'), 1.0])
    diagram_b.append([float('inf'), 2.0])

    diagram_a.append([float('inf'), 1.9])
    diagram_b.append([float('inf'), 1.1])

    diagram_a.append([10.0, 20.0])
    diagram_b.append([13.0, 19.0])

    correct_answer = 3.0

    assert (correct_answer) == (bottleneck_distance(diagram_a, diagram_b, 0))
    assert (correct_answer) == (bottleneck_distance(diagram_b, diagram_a, 0))
