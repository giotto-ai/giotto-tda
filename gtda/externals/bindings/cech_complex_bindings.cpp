/******************************************************************************
 * Description:      gudhi's cubical complex interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <Simplex_tree_interface.h>
#include <gudhi/Cech_complex.h>
#include <gudhi/Simplex_tree.h>

#include <iostream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace Gudhi {

namespace cech_complex {

class Cech_complex_interface {
 public:
  using Simplex_tree =
      Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
  using Filtration_value = Simplex_tree::Filtration_value;
  using Point_cloud = std::vector<std::vector<double>>;

  Cech_complex_interface(const Point_cloud& points,
                         Filtration_value max_radius) {
    cech_complex_ =
        new Cech_complex<Simplex_tree, Point_cloud>(points, max_radius);
  }

  ~Cech_complex_interface() {
    if (cech_complex_) delete cech_complex_;
  }

  void create_simplex_tree(Simplex_tree_interface<>* simplex_tree,
                           int dim_max) {
    if (cech_complex_) {
      cech_complex_->create_complex(*simplex_tree, dim_max);
      simplex_tree->initialize_filtration();
    }
  }

 private:
  Cech_complex<Simplex_tree, Point_cloud>* cech_complex_ = nullptr;
};
}  // namespace cech_complex
}  // namespace Gudhi

PYBIND11_MODULE(gtda_cech_complex, m) {
  using namespace pybind11::literals;
  using Simplex_tree =
      Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_fast_persistence>;
  py::class_<Gudhi::cech_complex::Cech_complex_interface>(
      m, "Cech_complex_interface")
      .def(py::init<Gudhi::cech_complex::Cech_complex_interface::Point_cloud,
                    Simplex_tree::Filtration_value>(),
           "points"_a, "max_radius"_a)
      .def("create_simplex_tree",
           &Gudhi::cech_complex::Cech_complex_interface::create_simplex_tree);
  m.doc() = "GUDHI Cech complex functions interfacing";
}
