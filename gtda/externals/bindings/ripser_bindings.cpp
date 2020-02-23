/******************************************************************************
 * Author:           Julián Burella Pérez
 * Description:      ripser's rips persistence interfacing with pybind11
 * License:          TBD
 *****************************************************************************/

#include "../ripser/ripser/ripser.cpp"

// PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_ripser, m) {
  m.doc() = "Ripser python interface";
  py::class_<ripserResults>(m, "ripserResults")
      .def_readwrite("births_and_deaths_by_dim",
                     &ripserResults::births_and_deaths_by_dim)
      .def_readwrite("cocycles_by_dim", &ripserResults::cocycles_by_dim)
      .def_readwrite("num_edges", &ripserResults::num_edges);

  m.def("rips_dm",
        [](std::vector<float> D, int N, int modulus, int dim_max,
           float threshold, int do_cocycles) {
          ripserResults ret =
              rips_dm(&D[0], N, modulus, dim_max, threshold, do_cocycles);
          return ret;
        },
        "ripser distance matrix");
  m.def("rips_dm_sparse",
        [](std::vector<int> I, std::vector<int> J, std::vector<float> V,
           int NEdges, int N, int modulus, int dim_max, float threshold,
           int do_cocycles) {
          ripserResults ret =
              rips_dm_sparse(&I[0], &J[0], &V[0], NEdges, N, modulus, dim_max,
                             threshold, do_cocycles);
          return ret;
        },
        "ripser sparse distance matrix");
}
