/******************************************************************************
 * Author:           Julián Burella Pérez
 * Description:      ripser's rips persistence interfacing with pybind11
 * License:          TBD
 *****************************************************************************/

#include <ripser.cpp>

// PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#if defined USE_COEFFICIENTS
PYBIND11_MODULE(gtda_ripser_coeff, m) {
#else
PYBIND11_MODULE(gtda_ripser, m) {
#endif

  using namespace pybind11::literals;
  m.doc() = "Ripser python interface";

  // Because `ripser` could have two different modules after compilation
  // It's necessary to add `py::module_local()` to prevent following issue:
  // ImportError: generic_type: type "ripserResults" is already registered!
  // When same python module imports gtda_ripser and gtda_ripser_coeff
  py::class_<ripserResults>(m, "ripserResults", py::module_local())
      .def_readwrite("births_and_deaths_by_dim",
                     &ripserResults::births_and_deaths_by_dim)
      .def_readwrite("cocycles_by_dim", &ripserResults::cocycles_by_dim)
      .def_readwrite("num_edges", &ripserResults::num_edges);

  m.def("rips_dm",
        [](py::array_t<float>& D, int N, int modulus, int dim_max,
           float threshold, int do_cocycles) {
          ripserResults ret = rips_dm((float*)D.request().ptr, N, modulus,
                                      dim_max, threshold, do_cocycles);
          return ret;
        },
        "D"_a, "N"_a, "modulus"_a, "dim_max"_a, "threshold"_a, "do_cocycles"_a,
        "ripser distance matrix");
  m.def("rips_dm_sparse",
        [](py::array_t<int>& I, py::array_t<int>& J, py::array_t<float>& V,
           int NEdges, int N, int modulus, int dim_max, float threshold,
           int do_cocycles) {
          ripserResults ret =
              rips_dm_sparse((int*)I.request().ptr, (int*)J.request().ptr,
                             (float*)V.request().ptr, NEdges, N, modulus,
                             dim_max, threshold, do_cocycles);
          return ret;
        },
        "I"_a, "J"_a, "V"_a, "NEdges"_a, "N"_a, "modulus"_a, "dim_max"_a,
        "threshold"_a, "do_cocycles"_a, "ripser sparse distance matrix");
}
