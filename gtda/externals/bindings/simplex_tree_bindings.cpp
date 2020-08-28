/******************************************************************************
 * Description:      gudhi's simplex tree interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include <iostream>

#include <Persistent_cohomology_interface.h>
#include <Simplex_tree_interface.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_simplex_tree, m) {
  // Simplex_tree_interface_full_featured
  using simplex_tree_interface_inst = Gudhi::Simplex_tree_interface<>;
  py::class_<simplex_tree_interface_inst>(
      m, "Simplex_tree_interface_full_featured")
      .def(py::init<>())
      .def("simplex_filtration",
           &simplex_tree_interface_inst::simplex_filtration)
      .def("assign_simplex_filtration",
           &simplex_tree_interface_inst::assign_simplex_filtration)
      .def("initialize_filtration",
           &simplex_tree_interface_inst::initialize_filtration)
      .def("num_vertices", &simplex_tree_interface_inst::num_vertices)
      .def("num_simplices",
           py::overload_cast<>(
               &simplex_tree_interface_inst::Simplex_tree::num_simplices))
      .def("set_dimension", &simplex_tree_interface_inst::set_dimension)
      .def("dimension",
           py::overload_cast<>(
               &simplex_tree_interface_inst::Simplex_tree::dimension))
      .def("upper_bound_dimension",
           &simplex_tree_interface_inst::upper_bound_dimension)
      .def("find_simplex", &simplex_tree_interface_inst::find_simplex)
      .def("insert_simplex_and_subfaces",
           py::overload_cast<
               const std::vector<simplex_tree_interface_inst::Vertex_handle>&,
               double>(
               &simplex_tree_interface_inst::insert_simplex_and_subfaces))
      .def("get_filtration",
           [](simplex_tree_interface_inst& self)
               -> std::vector<simplex_tree_interface_inst::Simplex_and_filtration> {
             std::vector<simplex_tree_interface_inst::Simplex_and_filtration> tmp;
             for (auto elem = self.get_filtration_iterator_begin();
                  elem != self.get_filtration_iterator_end(); elem++)
               tmp.push_back(self.get_simplex_and_filtration(*elem));
             return tmp;
           })
      .def("get_skeleton",
           [](simplex_tree_interface_inst& self, size_t dim)
               -> std::vector<
                   simplex_tree_interface_inst::Simplex_and_filtration> {
             std::vector<simplex_tree_interface_inst::Simplex_and_filtration>
                 tmp;
             for (auto elem = self.get_skeleton_iterator_begin(dim);
                  elem != self.get_skeleton_iterator_end(dim); elem++)
               tmp.push_back(self.get_simplex_and_filtration(*elem));
             return tmp;
           })
      .def("get_star", &simplex_tree_interface_inst::get_star)
      .def("get_cofaces", &simplex_tree_interface_inst::get_cofaces)
      .def("expansion", &simplex_tree_interface_inst::expansion)
      .def("remove_maximal_simplex",
           &simplex_tree_interface_inst::remove_maximal_simplex)
      .def("prune_above_filtration",
           &simplex_tree_interface_inst::prune_above_filtration)
      .def("make_filtration_non_decreasing",
           &simplex_tree_interface_inst::make_filtration_non_decreasing);
  // Simplex_tree_persistence_interface
  using Persistent_cohomology_interface_inst =
      Gudhi::Persistent_cohomology_interface<
          Gudhi::Simplex_tree<Gudhi::Simplex_tree_options_full_featured>>;
  py::class_<Persistent_cohomology_interface_inst>(
      m, "Simplex_tree_persistence_interface")
      .def(py::init<simplex_tree_interface_inst*, bool>())
      .def("compute_persistence",
           &Persistent_cohomology_interface_inst::compute_persistence)
      .def("get_persistence",
           &Persistent_cohomology_interface_inst::get_persistence)
      .def("betti_numbers",
           &Persistent_cohomology_interface_inst::betti_numbers)
      .def("persistent_betti_numbers",
           &Persistent_cohomology_interface_inst::persistent_betti_numbers)
      .def("intervals_in_dimension",
           &Persistent_cohomology_interface_inst::intervals_in_dimension)
      .def("persistence_pairs",
           &Persistent_cohomology_interface_inst::persistence_pairs)
      .def("write_output_diagram",
           &Persistent_cohomology_interface_inst::write_output_diagram);
  m.doc() = "GUDHI Simplex Tree functions interfacing";
}
