/******************************************************************************
 * Description:      gudhi's periodic cubical complex interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <Cubical_complex_interface.h>
#include <Persistent_cohomology_interface.h>
#include <gudhi/Bitmap_cubical_complex.h>
#include <gudhi/Bitmap_cubical_complex_base.h>
#include <gudhi/Bitmap_cubical_complex_periodic_boundary_conditions_base.h>
#include <iostream>

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_periodic_cubical_complex, m) {
  using Periodic_cubical_complex_inst =
      Gudhi::cubical_complex::Cubical_complex_interface<
          Gudhi::cubical_complex::
              Bitmap_cubical_complex_periodic_boundary_conditions_base<double>>;
  py::class_<Periodic_cubical_complex_inst>(
      m, "Periodic_cubical_complex_base_interface", py::buffer_protocol(),
      py::dynamic_attr())
      .def(py::init<const std::vector<unsigned>&, const std::vector<double>&,
                    const std::vector<bool>&>())
      .def(py::init<const std::string&>())
      .def("num_simplices", &Periodic_cubical_complex_inst::num_simplices)
      .def("dimension",
           py::overload_cast<>(&Periodic_cubical_complex_inst::dimension,
                               py::const_));

  using Persistent_cohomology_interface_inst =
      Gudhi::Persistent_cohomology_interface<
          Gudhi::cubical_complex::Cubical_complex_interface<
              Gudhi::cubical_complex::
                  Bitmap_cubical_complex_periodic_boundary_conditions_base<
                      double>>>;
  py::class_<Persistent_cohomology_interface_inst>(
      m, "Periodic_cubical_complex_persistence_interface")
      .def(py::init<
           Gudhi::cubical_complex::Cubical_complex_interface<
               Gudhi::cubical_complex::
                   Bitmap_cubical_complex_periodic_boundary_conditions_base<
                       double>>*,
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
  m.doc() = "GUDHI periocal cubical complex function interfacing";
}
