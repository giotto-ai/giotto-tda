/******************************************************************************
 * Created:          09/04/19
 * Description:      hera's bottleneck distance interfacing with pybind11
 * License:          Apache 2.0
 *****************************************************************************/

/* ssize_t is not standard C, it is a typedef from Posix.
 * Following solution is copy/pasted from solution found in
 * https://stackoverflow.com/a/35368387
 */
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <bottleneck/include/bottleneck.h>

// PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

double bottleneck_distance(std::vector<std::pair<double, double>>& dgm1,
                           std::vector<std::pair<double, double>>& dgm2,
                           double delta) {
  if (delta == 0.0)
    return hera::bottleneckDistExact(dgm1, dgm2);
  else
    return hera::bottleneckDistApprox(dgm1, dgm2, delta);
}

namespace py = pybind11;

PYBIND11_MODULE(gtda_bottleneck, m) {
  m.doc() = "bottleneck dionysus implementation";
  using namespace pybind11::literals;
  m.def("bottleneck_distance", &bottleneck_distance, "dgm1"_a, "dgm2"_a,
        py::arg("delta") = 0.01,
        "compute bottleneck distance between two persistence diagrams");
}
