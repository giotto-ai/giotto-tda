/******************************************************************************
 * Description:      gudhi's subsampling interfacing with pybind11
 * License:          GNU AGPLv3
 *****************************************************************************/

#include <Subsampling_interface.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_subsampling, m) {
  using namespace pybind11::literals;

  m.def("subsampling_n_farthest_points",
        py::overload_cast<const std::vector<std::vector<double>>&, unsigned>(
            &Gudhi::subsampling::subsampling_n_farthest_points),
        "points"_a, "nb_points"_a, "choose n farthest points");
  m.def("subsampling_n_farthest_points",
        py::overload_cast<const std::vector<std::vector<double>>&, unsigned,
                          unsigned>(
            &Gudhi::subsampling::subsampling_n_farthest_points),
        "points"_a, "nb_points"_a, "starting_point"_a,
        "choose n farthest points starting at a certain point");

  m.def("subsampling_n_farthest_points_from_file",
        py::overload_cast<const std::string&, unsigned>(
            &Gudhi::subsampling::subsampling_n_farthest_points_from_file),
        "off_file"_a, "nb_points"_a, "choose n farthest points from a file");
  m.def("subsampling_n_farthest_points_from_file",
        py::overload_cast<const std::string&, unsigned, unsigned>(
            &Gudhi::subsampling::subsampling_n_farthest_points_from_file),
        "off_file"_a, "nb_points"_a, "starting_point"_a,
        "choose n farthest points from a file starting at a certain point");

  m.def("subsampling_n_random_points",
        py::overload_cast<const std::vector<std::vector<double>>&, unsigned>(
            &Gudhi::subsampling::subsampling_n_random_points),
        "points"_a, "nb_points"_a, "pick n random points");
  m.def("subsampling_n_random_points_from_file",
        py::overload_cast<const std::string&, unsigned>(
            &Gudhi::subsampling::subsampling_n_random_points_from_file),
        "off_file"_a, "nb_points"_a, "pick n random points from a file");

  m.def("subsampling_sparsify_points",
        py::overload_cast<const std::vector<std::vector<double>>&, double>(
            &Gudhi::subsampling::subsampling_sparsify_points),
        "points"_a, "min_square_dist"_a, "sparsify point set");
  m.def("subsampling_sparsify_points_from_file",
        py::overload_cast<const std::string&, double>(
            &Gudhi::subsampling::subsampling_sparsify_points_from_file),
        "off_file"_a, "min_square_dist"_a, "sparsify point set from a file");

  m.doc() = "GUDHI subsampling functions interfacing";
}
