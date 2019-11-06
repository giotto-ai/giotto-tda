/******************************************************************************
 * Author:           Julián Burella Pérez
 * Description:      gudhi's rips complex interfacing with pybind11
 * License:          TBD
 *****************************************************************************/

#ifndef INCLUDE_RIPS_COMPLEX_INTERFACE_H_
#define INCLUDE_RIPS_COMPLEX_INTERFACE_H_

#include <gudhi/Rips_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Sparse_rips_complex.h>
#include <gudhi/distance_functions.h>

#include <iostream>
#include <string>
#include <utility>  // std::pair
#include <vector>

#include <boost/optional.hpp>

namespace Gudhi {

namespace rips_complex {

class Rips_complex_interface {
  using Point_d = std::vector<double>;
  using Distance_matrix =
      std::vector<std::vector<Simplex_tree_interface<>::Filtration_value>>;

 public:
  void init_points(const std::vector<std::vector<double>>& points,
                   double threshold) {
    rips_complex_.emplace(points, threshold, Gudhi::Euclidean_distance());
  }
  void init_matrix(const std::vector<std::vector<double>>& matrix,
                   double threshold) {
    rips_complex_.emplace(matrix, threshold);
  }

  void init_points_sparse(const std::vector<std::vector<double>>& points,
                          double threshold, double epsilon) {
    // sparse_rips_complex_.emplace(points, Gudhi::Euclidean_distance(),
    // epsilon,
    //                              -std::numeric_limits<double>::infinity(),
    //                              threshold);
  }
  void init_matrix_sparse(const std::vector<std::vector<double>>& matrix,
                          double threshold, double epsilon) {
    // sparse_rips_complex_.emplace(
    //     matrix, epsilon, -std::numeric_limits<double>::infinity(),
    //     threshold);
  }

  void create_simplex_tree(Simplex_tree_interface<>* simplex_tree,
                           int dim_max) {
    if (rips_complex_) {
      rips_complex_->create_complex(*simplex_tree, dim_max);
    }
    // else{
    //   // sparse_rips_complex_->create_complex(*simplex_tree, dim_max);
    // }
    simplex_tree->initialize_filtration();
  }

 private:
  // std::variant would work, but we don't require C++17 yet, and boost::variant
  // is not super convenient. Anyway, storing a graph would make more sense. Or
  // changing the interface completely so there is no such storage.
  boost::optional<Rips_complex<Simplex_tree_interface<>::Filtration_value>>
      rips_complex_;
  // boost::optional<Sparse_rips_complex<Simplex_tree_interface<>::Filtration_value>>
  // sparse_rips_complex_;
};

}  // namespace rips_complex

}  // namespace Gudhi

#endif  // INCLUDE_RIPS_COMPLEX_INTERFACE_H_
