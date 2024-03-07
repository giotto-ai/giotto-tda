from gtda.externals import PeriodicCubicalComplex


def test_empty_constructor():
    # Try to create an empty PeriodicCubicalComplex
    pcub = PeriodicCubicalComplex()
    assert pcub._PeriodicCubicalComplex__is_defined() is False
    assert pcub._PeriodicCubicalComplex__is_persistence_defined() is False


def test_non_existing_perseus_file_constructor():
    # Try to open a non existing file
    pcub = PeriodicCubicalComplex(perseus_file="does_not_exist.file")
    assert pcub._PeriodicCubicalComplex__is_defined() is False
    assert pcub._PeriodicCubicalComplex__is_persistence_defined() is False


def test_perseus_file_constructor():
    top_dimensional_cells = [1, 4, 6, 8, 20, 4, 7, 6, 5]
    pcub = PeriodicCubicalComplex(dimensions=[3, 3],
                                  top_dimensional_cells=top_dimensional_cells,
                                  periodic_dimensions=[True, False])
    assert pcub._PeriodicCubicalComplex__is_defined() is True
    assert pcub._PeriodicCubicalComplex__is_persistence_defined() is False
    assert pcub.dimension() == 2
    assert pcub.num_simplices() == 42


def test_perseus_file_persistence():
    top_dimensional_cells = \
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0]
    pcub = PeriodicCubicalComplex(dimensions=[3, 3, 3],
                                  top_dimensional_cells=top_dimensional_cells,
                                  periodic_dimensions=[False, False, False])
    diag = pcub.persistence(homology_coeff_field=3, min_persistence=0)
    assert pcub._PeriodicCubicalComplex__is_defined() is True
    assert pcub._PeriodicCubicalComplex__is_persistence_defined() is True
    assert diag == [(2, (0.0, 100.0)), (0, (0.0, float('inf')))]
