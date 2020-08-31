/******************************************************************************
 * Description:      gudhi's collapser interfacing with pybind11
 * License:          AGPL3
 *****************************************************************************/

#include <gudhi/Flag_complex_edge_collapser.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/* GUDHI Collapser required types */
using Filtration_value = float;
using Vertex_handle = uint32_t;
using Filtered_edge =
    std::tuple<Vertex_handle, Vertex_handle, Filtration_value>;
using Filtered_edge_list = std::vector<Filtered_edge>;

/* Sparse matrix input types */
using Sparse_matrix = Eigen::SparseMatrix<Filtration_value>;
using triplet_vec = Eigen::Triplet<Filtration_value>;

/* COO data input types */
using Row_idx = std::vector<Vertex_handle>;
using Col_idx = std::vector<Vertex_handle>;
using Filtration_values = std::vector<Filtration_value>;
using COO_data = std::tuple<Row_idx, Col_idx, Filtration_values>;

/* Dense matrix input types */
using Distance_matrix = std::vector<std::vector<Filtration_value>>;

/* Generates a sparse matrix from a filtered edge list
 * This function is called after computing edge collapse
 * At the moment this function is unused because Eigen only manages
 * CSR sparse format, but in the case of giotto-tda we need COO format
 */
static Sparse_matrix generate_sparse_matrix(Filtered_edge_list& collapsed_edges,
                                            size_t size) {
  std::vector<triplet_vec> triplets;
  /* Create triplets to efficiently create a return matrix */
  for (auto& t : collapsed_edges) {
    triplets.push_back(
        triplet_vec(std::get<0>(t), std::get<1>(t), std::get<2>(t)));
    std::cout << std::get<0>(t) << ", " << std::get<1>(t) << " : "
              << std::get<2>(t) << "\n";
  }

  Sparse_matrix mat(size, size);
  mat.setFromTriplets(triplets.begin(), triplets.end());

  return mat;
}

/* Generates COO sparse matrix data from a filtered edge list
 * This function is called after computing edge collapse
 */
static COO_data gen_coo_matrix(Filtered_edge_list& collapsed_edges) {
  Row_idx row;
  Col_idx col;
  Filtration_values data;

  /* allocate memory beforehand */
  row.reserve(collapsed_edges.size());
  col.reserve(collapsed_edges.size());
  data.reserve(collapsed_edges.size());

  for (auto& t : collapsed_edges) {
    row.push_back(std::get<0>(t));
    col.push_back(std::get<1>(t));
    data.push_back(std::get<2>(t));
  }

  return COO_data(row, col, data);
}

PYBIND11_MODULE(gtda_collapser, m) {
  using namespace pybind11::literals;

  m.doc() = "Collapser bindings for GUDHI implementation";
  m.def("flag_complex_collapse_edges",
        [](Sparse_matrix& graph_) {
          Filtered_edge_list graph;

          /* Convert from sparse format to Filtered_edge_list */
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

          return gen_coo_matrix(vec_triples);
        },
        "sparse_matrix"_a,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");

  m.def("flag_complex_collapse_edges",
        [](Row_idx& row, Col_idx& col, Filtration_values& data) {
          Filtered_edge_list graph;

          /* Convert from COO input format to Filtered_edge_list */
          int size = data.size();
          for (size_t k = 0; k < size; ++k)
            graph.push_back(Filtered_edge(row[k], col[k], data[k]));

          /* Start collapser */
          auto vec_triples =
              Gudhi::collapse::flag_complex_collapse_edges(graph);

          return gen_coo_matrix(vec_triples);
        },
        "row"_a, "column"_a, "data"_a,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");

  m.def("flag_complex_collapse_edges",
        [](Distance_matrix& dm) {
          Filtered_edge_list graph;

          /* Convert from dense format to Filtered edge list */
          for (size_t i = 0; i < dm.size(); i++)
            for (size_t j = 0; j < dm[i].size(); j++)
              if (j > i) graph.push_back(Filtered_edge(i, j, dm[i][j]));

          /* Start collapser */
          auto vec_triples =
              Gudhi::collapse::flag_complex_collapse_edges(graph);

          return gen_coo_matrix(vec_triples);
        },
        "dense_matrix"_a,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");
}
