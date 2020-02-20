/******************************************************************************
 * Description:      gudhi's witness complex interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <Simplex_tree_interface.h>
#include <Witness_complex_interface.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_witness_complex, m) {
  using simplex_tree_interface_inst = Gudhi::Simplex_tree_interface<>;
  using witness_interface_inst =
      Gudhi::witness_complex::Witness_complex_interface;
  py::class_<witness_interface_inst>(m, "Witness_complex_interface")
      .def(py::init<
           const std::vector<std::vector<std::pair<std::size_t, double>>>&>())
      .def("create_simplex_tree",
           py::overload_cast<simplex_tree_interface_inst*, double>(
               &witness_interface_inst::create_simplex_tree))
      .def("create_simplex_tree",
           py::overload_cast<simplex_tree_interface_inst*, double, std::size_t>(
               &witness_interface_inst::create_simplex_tree));
  m.doc() = "GUDHI Witness Complex functions interfacing";
}
