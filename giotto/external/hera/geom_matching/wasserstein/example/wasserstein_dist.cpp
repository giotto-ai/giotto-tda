/*

Copyright (c) 2015, M. Kerber, D. Morozov, A. Nigmetov
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
(Enhancements) to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to copyright holder,
without imposing a separate written license agreement for such Enhancements,
then you hereby grant the following license: a  non-exclusive, royalty-free
perpetual license to install, use, modify, prepare derivative works, incorporate
into other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.

  */

#include <iostream>
#include <locale>
#include <iomanip>
#include <vector>

#include "opts/opts.h"

//#define LOG_AUCTION

//#include "auction_runner_fr.h"
//#include "auction_runner_fr.hpp"

#include "wasserstein.h"

// any container of pairs of doubles can be used,
// we use vector in this example.

int main(int argc, char* argv[])
{
    using PairVector = std::vector<std::pair<double, double>>;
    PairVector diagramA, diagramB;

    hera::AuctionParams<double> params;
    params.max_num_phases = 800;

    opts::Options ops(argc, argv);
    ops >> opts::Option('q', "degree", params.wasserstein_power, "Wasserstein degree")
        >> opts::Option('d', "error", params.delta, "Relative error")
        >> opts::Option('p', "internal-p", params.internal_p, "Internal norm")
        >> opts::Option("initial-epsilon", params.initial_epsilon, "Initial epsilon")
        >> opts::Option("epsilon-factor", params.epsilon_common_ratio, "Epsilon factor")
        >> opts::Option("max-bids-per-round", params.max_bids_per_round, "Maximal number of bids per round")
        >> opts::Option('m', "max-rounds", params.max_num_phases, "Maximal number of iterations");


    bool print_relative_error = ops >> opts::Present('e', "--print-error", "Print real relative error");

    params.tolerate_max_iter_exceeded  = ops >> opts::Present('t', "tolerate", "Suppress max-iterations-exceeded error and print the best result.");

    std::string dgm_fname_1, dgm_fname_2;
    bool dgm_1_given = (ops >> opts::PosOption(dgm_fname_1));
    bool dgm_2_given = (ops >> opts::PosOption(dgm_fname_2));

    //std::cout << "q = " << params.wasserstein_power << ", delta = " << params.delta << ", p = " << params.internal_p << ", max_round = " << params.max_num_phases <<  std::endl;
    //std::cout << "print relative error: " << print_relative_error << std::endl;
    //std::cout << "dgm1: " << dgm_fname_1 << std::endl;
    //std::cout << "dgm2: " << dgm_fname_2 << std::endl;

    if (not dgm_1_given or not dgm_2_given) {
        std::cerr << "Usage: " << argv[0] << " file1 file2 " << std::endl;
        std::cerr << "compute Wasserstein distance between persistence diagrams in file1 and file2.\n";
        std::cerr << ops << std::endl;
        return 1;
    }

    if (ops >> opts::Present('h', "help", "show help message")) {
        std::cout << "Usage: " << argv[0] << " file1 file2 " << std::endl;
        std::cout << "compute Wasserstein distance between persistence diagrams in file1 and file2.\n";
        std::cout << ops << std::endl;
    }

    if (!hera::read_diagram_point_set<double, PairVector>(dgm_fname_1, diagramA)) {
        std::exit(1);
    }

    if (!hera::read_diagram_point_set(dgm_fname_2, diagramB)) {
        std::exit(1);
    }

    if (params.wasserstein_power < 1.0) {
        std::cerr << "Wasserstein_degree was \"" << params.wasserstein_power << "\", must be a number >= 1.0. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    if (params.wasserstein_power == 1.0) {
        hera::remove_duplicates<double>(diagramA, diagramB);
    }

    //default relative error:  1%
    if ( params.delta <= 0.0) {
        std::cerr << "relative error was \"" << params.delta << "\", must be a number > 0.0. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    // default for internal metric is l_infinity
    if (std::isinf(params.internal_p)) {
        params.internal_p = hera::get_infinity<double>();
    }


    if (not hera::is_p_valid_norm<double>(params.internal_p)) {
        std::cerr << "internal-p was \"" << params.internal_p << "\", must be a number >= 1.0 or inf. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    // if you want to specify initial value for epsilon and the factor
    // for epsilon-scaling
    if (params.initial_epsilon < 0.0) {
        std::cerr << "initial-epsilon was \"" << params.initial_epsilon << "\", must be a non-negative number. Cannot proceed." << std::endl;
        std::exit(1);
    }

    if (params.epsilon_common_ratio <= 1.0 and params.epsilon_common_ratio != 0.0) {
        std::cerr << "The 7th argument (epsilon factor) was \"" << params.epsilon_common_ratio << "\", must be a number greater than 1. Cannot proceed." << std::endl;
        std::exit(1);
    }

    if (params.max_bids_per_round == 0)
        params.max_bids_per_round = std::numeric_limits<decltype(params.max_bids_per_round)>::max();


    std::string log_filename_prefix = ( 11 <= argc ) ? argv[10] : "";


#ifdef LOG_AUCTION
    spdlog::set_level(spdlog::level::info);
#endif

    double res = hera::wasserstein_dist(diagramA, diagramB, params, log_filename_prefix);

    std::cout << std::setprecision(15) << res << std::endl;
    if (print_relative_error)
        std::cout << "Relative error: " << params.final_relative_error << std::endl;

    return 0;

}
