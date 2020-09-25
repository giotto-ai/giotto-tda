/******************************************************************************
 * Description:      gudhi's collapser interfacing with pybind11
 * License:          AGPL3
 *****************************************************************************/

#include <gudhi/Flag_complex_edge_collapser.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/* GUDHI Collapser required types */
using Filtration_value = float;
using Vertex_handle = int32_t;
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
using Distance_matrix_np = py::array_t<Filtration_value>;

/* constants */
const Filtration_value filtration_max =
    std::numeric_limits<Filtration_value>::infinity();

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
  m.def("flag_complex_collapse_edges_sparse",
        [](Sparse_matrix& sm, Filtration_value thresh = filtration_max) {
          Filtered_edge_list graph;

          /* Convert from sparse format to Filtered_edge_list */
          /* Applying threshold to the input data */
          int size = sm.outerSize();
          for (size_t k = 0; k < size; ++k)
            for (Eigen::SparseMatrix<Filtration_value>::InnerIterator it(sm, k);
                 it; ++it) {
              if (it.value() <= thresh)
                graph.push_back(Filtered_edge(it.row(), it.col(), it.value()));
            }

          /* Start collapser */
          auto vec_triples =
              Gudhi::collapse::flag_complex_collapse_edges(graph);

          return gen_coo_matrix(vec_triples);
        },
        "sm"_a, "thresh"_a = filtration_max,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");

  m.def("flag_complex_collapse_edges_coo",
        [](py::array_t<Vertex_handle>& row_, py::array_t<Vertex_handle>& col_,
           py::array_t<Filtration_value>& data_,
           Filtration_value thresh = filtration_max) {
          Filtered_edge_list graph;

          Vertex_handle* row = (Vertex_handle*)row_.request().ptr;
          Vertex_handle* col = (Vertex_handle*)col_.request().ptr;
          Filtration_value* data = (Filtration_value*)data_.request().ptr;

          /* Convert from COO input format to Filtered_edge_list */
          /* Applying threshold to the input data */
          int size = data_.size();
          for (size_t k = 0; k < size; ++k)
            if (data[k] <= thresh)
              graph.push_back(Filtered_edge(row[k], col[k], data[k]));

          /* Start collapser */
          auto vec_triples =
              Gudhi::collapse::flag_complex_collapse_edges(graph);

          return gen_coo_matrix(vec_triples);
        },
        "row"_a, "column"_a, "data"_a, "thresh"_a = filtration_max,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");

  m.def("flag_complex_collapse_edges_dense",
        [](Distance_matrix_np& dm, Filtration_value thresh = filtration_max) {
          Filtered_edge_list graph;

          /* Convert from dense format to Filtered edge list */
          /* Applying threshold to the input data */
          for (size_t i = 0; i < dm.shape(0); i++)
            for (size_t j = 0; j < dm.shape(1); j++)
              if (j > i && (*(dm.data(i, j)) <= thresh))
                graph.push_back(Filtered_edge(i, j, *(dm.data(i, j))));

          /* Start collapser */
          auto vec_triples =
              Gudhi::collapse::flag_complex_collapse_edges(graph);

          return gen_coo_matrix(vec_triples);
        },
        "dm"_a, "thresh"_a = filtration_max,
        "Implicitly constructs a flag complex from edges, "
        "collapses edges while preserving the persistent homology");
}
