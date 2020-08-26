/******************************************************************************
 * Description:      gudhi's persistent cohomology interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <iostream>
#include <Persistent_cohomology_interface.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cubical_complex_bindings.cpp"

namespace py = pybind11;

PYBIND11_MODULE(gtda_persistent_cohomology, m) {
  using Persistent_cohomology_interface_inst =
      Gudhi::Persistent_cohomology_interface<
          Gudhi::cubical_complex::Cubical_complex_interface<>>;
  py::class_<Persistent_cohomology_interface_inst>(
      m, "Persistent_cohomology_interface")
      .def(py::init<Gudhi::cubical_complex::Cubical_complex_interface<>*>())
      .def(py::init<Gudhi::cubical_complex::Cubical_complex_interface<>*,
                    bool>())
      .def("compute_persistence",
           &Persistent_cohomology_interface_inst::compute_persistence)
      .def("get_persistence",
           &Persistent_cohomology_interface_inst::get_persistence)
      .def("betti_numbers",
           &Persistent_cohomology_interface_inst::betti_numbers)
      .def("persistent_betti_numbers",
           &Persistent_cohomology_interface_inst::persistent_betti_numbers)
      .def("intervals_in_dimension",
           &Persistent_cohomology_interface_inst::intervals_in_dimension);
  m.doc() = "GUDHI persistent homology interfacing";
}
