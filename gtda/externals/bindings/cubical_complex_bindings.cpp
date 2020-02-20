/******************************************************************************
 * Description:      gudhi's cubical complex interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <gudhi/Bitmap_cubical_complex.h>
#include <gudhi/Bitmap_cubical_complex_base.h>
#include <gudhi/Bitmap_cubical_complex_periodic_boundary_conditions_base.h>
#include <Cubical_complex_interface.h>

#include <iostream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_cubical_complex, m) {
  using namespace pybind11::literals;
  using Cubical_complex_interface_inst =
      Gudhi::cubical_complex::Cubical_complex_interface<>;
  using Bitmap_cubical_complex_inst =
      Gudhi::Cubical_complex::Bitmap_cubical_complex<
          Gudhi::Cubical_complex::Bitmap_cubical_complex_base<double>>;
  py::class_<Bitmap_cubical_complex_inst, Cubical_complex_interface_inst>(
      m, "Cubical_complex_interface", py::buffer_protocol(), py::dynamic_attr())
      .def(py::init<const std::vector<unsigned>&, const std::vector<double>&>(),
           "dimensions"_a, "top_dimensional_cells"_a)
      .def(py::init<const std::vector<unsigned>&, const std::vector<double>&,
                    const std::vector<bool>&>(),
           "dimensions"_a, "top_dimensional_cells"_a, "periodic_dimensions"_a)
      .def(py::init<const std::string&>(), "perseus_file"_a)
      .def("num_simplices", &Cubical_complex_interface_inst::num_simplices)
      .def("dimension",
           py::overload_cast<>(&Cubical_complex_interface_inst::dimension,
                               py::const_));
  m.doc() = "GUDHI cubical complex function interfacing";
}
