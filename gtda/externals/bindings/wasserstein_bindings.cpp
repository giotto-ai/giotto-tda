/******************************************************************************
 * Description:      hera's wasserstein distance interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#include "../hera/wasserstein/wasserstein.h"

// PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

double wasserstein_distance(const std::vector<std::pair<double, double>>& dgm1,
                            const std::vector<std::pair<double, double>>& dgm2,
                            int q, double delta, double internal_p,
                            double initial_eps, double eps_factor) {
  hera::AuctionParams<double> params;
  params.wasserstein_power = q;
  params.delta = delta;
  params.internal_p = internal_p;

  if (initial_eps != 0) params.initial_epsilon = initial_eps;

  if (eps_factor != 0.) params.epsilon_common_ratio = eps_factor;

  return hera::wasserstein_dist(dgm1, dgm2, params);
}

namespace py = pybind11;

PYBIND11_MODULE(gtda_wasserstein, m) {
  m.doc() = "wasserstein dionysus implementation";
  using namespace pybind11::literals;
  m.def("wasserstein_distance", &wasserstein_distance, "dgm1"_a, "dgm2"_a,
        py::arg("q") = 2, py::arg("delta") = .01,
        py::arg("internal_p") = hera::get_infinity<double>(),
        py::arg("initial_eps") = 0., py::arg("eps_factor") = 0.,
        "compute Wasserstein distance between two persistence diagrams");
}
