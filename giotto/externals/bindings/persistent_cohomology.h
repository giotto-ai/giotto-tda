/******************************************************************************
 * Author:           Julián Burella Pérez
 * Description:      gudhi's persistent cohomology interfacing with pybind11
 * License:          TBD
 *****************************************************************************/

#ifndef INCLUDE_PERSISTENT_COHOMOLOGY_INTERFACE_H_
#define INCLUDE_PERSISTENT_COHOMOLOGY_INTERFACE_H_

#include <gudhi/Persistent_cohomology.h>
#include <iostream>

#include <algorithm>  // for sort
#include <utility>    // for std::pair
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Gudhi {

template <class FilteredComplex>
class Persistent_cohomology_interface
    : public persistent_cohomology::Persistent_cohomology<
          FilteredComplex, persistent_cohomology::Field_Zp> {
 private:
  /*
   * Compare two intervals by dimension, then by length.
   */
  struct cmp_intervals_by_dim_then_length {
    explicit cmp_intervals_by_dim_then_length(FilteredComplex* sc) : sc_(sc) {}

    template <typename Persistent_interval>
    bool operator()(const Persistent_interval& p1,
                    const Persistent_interval& p2) {
      if (sc_->dimension(get<0>(p1)) == sc_->dimension(get<0>(p2)))
        return (sc_->filtration(get<1>(p1)) - sc_->filtration(get<0>(p1)) >
                sc_->filtration(get<1>(p2)) - sc_->filtration(get<0>(p2)));
      else
        return (sc_->dimension(get<0>(p1)) > sc_->dimension(get<0>(p2)));
    }
    FilteredComplex* sc_;
  };

 public:
  Persistent_cohomology_interface(FilteredComplex* stptr)
      : persistent_cohomology::Persistent_cohomology<
            FilteredComplex, persistent_cohomology::Field_Zp>(*stptr),
        stptr_(stptr) {}

  Persistent_cohomology_interface(FilteredComplex* stptr,
                                  bool persistence_dim_max)
      : persistent_cohomology::Persistent_cohomology<
            FilteredComplex, persistent_cohomology::Field_Zp>(
            *stptr, persistence_dim_max),
        stptr_(stptr) {}

  std::vector<std::pair<int, std::pair<double, double>>> get_persistence(
      int homology_coeff_field, double min_persistence) {
    persistent_cohomology::Persistent_cohomology<
        FilteredComplex, persistent_cohomology::Field_Zp>::
        init_coefficients(homology_coeff_field);
    persistent_cohomology::Persistent_cohomology<
        FilteredComplex, persistent_cohomology::Field_Zp>::
        compute_persistent_cohomology(min_persistence);

    // Custom sort and output persistence
    cmp_intervals_by_dim_then_length cmp(stptr_);
    auto persistent_pairs = persistent_cohomology::Persistent_cohomology<
        FilteredComplex,
        persistent_cohomology::Field_Zp>::get_persistent_pairs();
    std::sort(std::begin(persistent_pairs), std::end(persistent_pairs), cmp);

    std::vector<std::pair<int, std::pair<double, double>>> persistence;
    for (auto pair : persistent_pairs) {
      persistence.push_back(
          std::make_pair(stptr_->dimension(get<0>(pair)),
                         std::make_pair(stptr_->filtration(get<0>(pair)),
                                        stptr_->filtration(get<1>(pair)))));
    }
    return persistence;
  }

  std::vector<std::pair<std::vector<int>, std::vector<int>>>
  persistence_pairs() {
    auto pairs = persistent_cohomology::Persistent_cohomology<
        FilteredComplex,
        persistent_cohomology::Field_Zp>::get_persistent_pairs();

    std::vector<std::pair<std::vector<int>, std::vector<int>>>
        persistence_pairs;
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

#endif
