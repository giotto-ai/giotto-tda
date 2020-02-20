/******************************************************************************
 * Description:      gudhi's cubical complex interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <Rips_complex_interface.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_sparse_rips_complex, m) {
  py::class_<Gudhi::rips_complex::Rips_complex_interface>(
      m, "Rips_complex_interface")
      .def(py::init<>())
      .def("init_points",
           &Gudhi::rips_complex::Rips_complex_interface::init_points)
      .def("init_matrix",
           &Gudhi::rips_complex::Rips_complex_interface::init_matrix)
      .def("init_points_sparse",
           &Gudhi::rips_complex::Rips_complex_interface::init_points_sparse)
      .def("init_matrix_sparse",
           &Gudhi::rips_complex::Rips_complex_interface::init_matrix_sparse)
      .def("create_simplex_tree",
           &Gudhi::rips_complex::Rips_complex_interface::create_simplex_tree);
  m.doc() = "GUDHI Sparse Rips Complex functions interfacing";
}
