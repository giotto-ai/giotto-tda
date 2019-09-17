/******************************************************************************
* Author:           Julián Burella Pérez
* Description:      gudhi's persistent cohomology interfacing with pybind11
* License:          TBD
*****************************************************************************/

#include <iostream>
#include <gudhi/Persistent_cohomology.h>

#include <vector>
#include <utility>  // for std::pair
#include <algorithm>  // for sort

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cubical_complex_bindings.cpp"

namespace Gudhi {

template<class FilteredComplex>
class Persistent_cohomology_interface : public
persistent_cohomology::Persistent_cohomology<FilteredComplex, persistent_cohomology::Field_Zp> {
 private:
  /*
   * Compare two intervals by dimension, then by length.
   */
  struct cmp_intervals_by_dim_then_length {
    explicit cmp_intervals_by_dim_then_length(FilteredComplex * sc)
        : sc_(sc) { }

    template<typename Persistent_interval>
    bool operator()(const Persistent_interval & p1, const Persistent_interval & p2) {
      if (sc_->dimension(get < 0 > (p1)) == sc_->dimension(get < 0 > (p2)))
        return (sc_->filtration(get < 1 > (p1)) - sc_->filtration(get < 0 > (p1))
                > sc_->filtration(get < 1 > (p2)) - sc_->filtration(get < 0 > (p2)));
      else
        return (sc_->dimension(get < 0 > (p1)) > sc_->dimension(get < 0 > (p2)));
    }
    FilteredComplex* sc_;
  };

 public:
  Persistent_cohomology_interface(FilteredComplex* stptr)
      : persistent_cohomology::Persistent_cohomology<FilteredComplex, persistent_cohomology::Field_Zp>(*stptr),
      stptr_(stptr) { }

  Persistent_cohomology_interface(FilteredComplex* stptr, bool persistence_dim_max)
      : persistent_cohomology::Persistent_cohomology<FilteredComplex,
          persistent_cohomology::Field_Zp>(*stptr, persistence_dim_max),
        stptr_(stptr) { }

  std::vector<std::pair<int, std::pair<double, double>>> get_persistence(int homology_coeff_field,
                                                                         double min_persistence) {
    persistent_cohomology::Persistent_cohomology<FilteredComplex,
      persistent_cohomology::Field_Zp>::init_coefficients(homology_coeff_field);
    persistent_cohomology::Persistent_cohomology<FilteredComplex,
      persistent_cohomology::Field_Zp>::compute_persistent_cohomology(min_persistence);

    // Custom sort and output persistence
    cmp_intervals_by_dim_then_length cmp(stptr_);
    auto persistent_pairs = persistent_cohomology::Persistent_cohomology<FilteredComplex,
      persistent_cohomology::Field_Zp>::get_persistent_pairs();
    std::sort(std::begin(persistent_pairs), std::end(persistent_pairs), cmp);

    std::vector<std::pair<int, std::pair<double, double>>> persistence;
    for (auto pair : persistent_pairs) {
      persistence.push_back(std::make_pair(stptr_->dimension(get<0>(pair)),
                                           std::make_pair(stptr_->filtration(get<0>(pair)),
                                                          stptr_->filtration(get<1>(pair)))));
    }
    return persistence;
  }

  std::vector<std::pair<std::vector<int>, std::vector<int>>> persistence_pairs() {
    auto pairs = persistent_cohomology::Persistent_cohomology<FilteredComplex,
      persistent_cohomology::Field_Zp>::get_persistent_pairs();

    std::vector<std::pair<std::vector<int>, std::vector<int>>> persistence_pairs;
    persistence_pairs.reserve(pairs.size());
    for (auto pair : pairs) {
      std::vector<int> birth;
      if (get<0>(pair) != stptr_->null_simplex()) {
        for (auto vertex : stptr_->simplex_vertex_range(get<0>(pair))) {
          birth.push_back(vertex);
        }
      }

      std::vector<int> death;
      if (get<1>(pair) != stptr_->null_simplex()) {
        for (auto vertex : stptr_->simplex_vertex_range(get<1>(pair))) {
          death.push_back(vertex);
        }
      }

      persistence_pairs.push_back(std::make_pair(birth, death));
    }
    return persistence_pairs;
  }

 private:
  // A copy
  FilteredComplex* stptr_;
};

}  // namespace Gudhi

namespace py = pybind11;

PYBIND11_MODULE(giotto_persistent_cohomology, m) {
    using Persistent_cohomology_interface_inst = Gudhi::Persistent_cohomology_interface<Cubical_complex_interface<>>;
    py::class_<Persistent_cohomology_interface_inst>(m, "Persistent_cohomology_interface")
        .def(py::init<Cubical_complex_interface<>*>())
        .def(py::init<Cubical_complex_interface<>*, bool>())
        .def("get_persistence", &Persistent_cohomology_interface_inst::get_persistence)
        .def("betti_numbers", &Persistent_cohomology_interface_inst::betti_numbers)
        .def("persistent_betti_numbers", &Persistent_cohomology_interface_inst::persistent_betti_numbers)
        .def("intervals_in_dimension", &Persistent_cohomology_interface_inst::intervals_in_dimension)
        ;
    m.doc() = "GUDHI persistant homology interfacing";
}
