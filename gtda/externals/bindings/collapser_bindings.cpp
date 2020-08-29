/******************************************************************************
 * Description:      gudhi's collapser interfacing with pybind11
 * License:          AGPL3
 *****************************************************************************/

#include <gudhi/Flag_complex_edge_collapser.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/* GUDHI Collapser require types */
using Filtration_value = float;
using Vertex_handle = uint32_t;
using Filtered_edge =
    std::tuple<Vertex_handle, Vertex_handle, Filtration_value>;
using Filtered_edge_list = std::vector<Filtered_edge>;

/* sparse matrix input types */
using Sparse_matrix = Eigen::SparseMatrix<Filtration_value>;
using triplet_vec = Eigen::Triplet<Filtration_value>;

/* dense matrix input types */
using Distance_matrix = std::vector<std::vector<Filtration_value>>;

/* generates a sparse matrix from a filtered edges list
 * This function is called after computing edges collapsing
 */
static Sparse_matrix generate_sparse_matrix(
    Filtered_edge_list& collapsed_edges, int size) {
  std::vector<triplet_vec> triplets;
  /* Create triplets to efficiently create a return matrix */
  for (auto& t : collapsed_edges) {
    triplets.push_back(
        triplet_vec(std::get<0>(t), std::get<1>(t), std::get<2>(t)));
  }

  Sparse_matrix mat(size, size);
  mat.setFromTriplets(triplets.begin(), triplets.end());

  return mat;
}

PYBIND11_MODULE(gtda_collapser, m) {
  using namespace pybind11::literals;

  m.doc() = "Collapser bindings for Gudhi  implementation";
  m.def("flag_complex_collapse_edges",
        [](Sparse_matrix& graph_) {
          Filtered_edge_list graph;

          /* Convert from Sparse format to Filtered edge list */
          int size = graph_.outerSize();
          for (size_t k = 0; k < size; ++k)
            for (Eigen::SparseMatrix<Filtration_value>::InnerIterator it(graph_,
                                                                         k);
                 it; ++it) {
              graph.push_back(Filtered_edge(it.row(), it.col(), it.value()));
            }

          /* Start collapser */
          auto vec_triples =
              Gudhi::collapse::flag_complex_collapse_edges(graph);
          return generate_sparse_matrix(vec_triples, size);
        },
        "sparse_matrix"_a,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");

  m.def("flag_complex_collapse_edges",
        [](Distance_matrix& dm) {
          Filtered_edge_list graph;
          std::vector<triplet_vec> triplets;

          /* Convert from Sparse format to Filtered edge list */
          for (size_t i = 0; i < dm.size(); i++)
            for (size_t j = 0; j < dm[i].size(); j++)
              if (j > i) graph.push_back(Filtered_edge(i, j, dm[i][j]));

          /* Start collapser */
          auto vec_triples =
              Gudhi::collapse::flag_complex_collapse_edges(graph);

          return generate_sparse_matrix(vec_triples, dm.size());
        },
        "dense_matrix"_a,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");
}
