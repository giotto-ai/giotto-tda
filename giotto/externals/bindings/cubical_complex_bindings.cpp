/******************************************************************************
* Author:           Julián Burella Pérez
* Description:      gudhi's cubical complex interfacing with pybind11
* License:          TBD
*****************************************************************************/

#include <gudhi/Bitmap_cubical_complex.h>
#include <gudhi/Bitmap_cubical_complex_base.h>
#include <gudhi/Bitmap_cubical_complex_periodic_boundary_conditions_base.h>

#include <iostream>
#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template<typename CubicalComplexOptions = Gudhi::Cubical_complex::Bitmap_cubical_complex_base<double>>
class Cubical_complex_interface : public Gudhi::Cubical_complex::Bitmap_cubical_complex<CubicalComplexOptions> {
 public:
  Cubical_complex_interface(const std::vector<unsigned>& dimensions,
                            const std::vector<double>& top_dimensional_cells)
  : Gudhi::cubical_complex::Bitmap_cubical_complex<CubicalComplexOptions>(dimensions, top_dimensional_cells) {
  }

  Cubical_complex_interface(const std::vector<unsigned>& dimensions,
                            const std::vector<double>& top_dimensional_cells,
                            const std::vector<bool>& periodic_dimensions)
  : Gudhi::cubical_complex::Bitmap_cubical_complex<CubicalComplexOptions>(dimensions, top_dimensional_cells, periodic_dimensions) {
  }

  Cubical_complex_interface(const std::string& perseus_file)
  : Gudhi::cubical_complex::Bitmap_cubical_complex<CubicalComplexOptions>(perseus_file.c_str()) {
  }
};

namespace py = pybind11;

PYBIND11_MODULE(giotto_cubical_complex, m) {
    using Cubical_complex_interface_inst = Cubical_complex_interface<>;
    using Bitmap_cubical_complex_inst = Gudhi::Cubical_complex::Bitmap_cubical_complex<Gudhi::Cubical_complex::Bitmap_cubical_complex_base<double>>;
    // py::class_<Bitmap_cubical_complex_inst>(m, "Bitmap_cubical_complex");
    py::class_<Bitmap_cubical_complex_inst, Cubical_complex_interface_inst>(m, "Cubical_complex_interface",
            py::buffer_protocol(), py::dynamic_attr())
        .def(py::init<const std::vector<unsigned>&, const std::vector<double>&>())
        .def(py::init<const std::vector<unsigned>&, const std::vector<double>&, const std::vector<bool>&>())
        .def(py::init<const std::string&>())
        .def("num_simplices", &Cubical_complex_interface_inst::num_simplices)
        .def("dimension", py::overload_cast<>(&Cubical_complex_interface_inst::dimension, py::const_))
        ;
    m.doc() = "GUDHI cubical complex function interfacing";
}
