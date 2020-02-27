/******************************************************************************
 * Description:      gudhi's strong witness complex interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <Simplex_tree_interface.h>
#include <Strong_witness_complex_interface.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_strong_witness_complex, m) {
  using simplex_tree_interface_inst = Gudhi::Simplex_tree_interface<>;
  using strong_witness_interface_inst =
      Gudhi::witness_complex::Strong_witness_complex_interface;
  py::class_<strong_witness_interface_inst>(m,
                                            "Strong_witness_complex_interface")
      .def(py::init<
           const std::vector<std::vector<std::pair<std::size_t, double>>>&>())
      .def("create_simplex_tree",
           py::overload_cast<simplex_tree_interface_inst*, double>(
               &strong_witness_interface_inst::create_simplex_tree))
      .def("create_simplex_tree",
           py::overload_cast<simplex_tree_interface_inst*, double, std::size_t>(
               &strong_witness_interface_inst::create_simplex_tree));
  m.doc() = "GUDHI Strong Witness Complex functions interfacing";
}
