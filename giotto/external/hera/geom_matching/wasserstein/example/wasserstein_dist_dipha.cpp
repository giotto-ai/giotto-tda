/*

Copyright (c) 2015, M. Kerber, D. Morozov, A. Nigmetov
Copyright (c) 2018, G. Spreemann
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

    if (argc < 4 ) {
        std::cerr << "Usage: " << argv[0] << " file1 file2 ph_dim [wasserstein_degree] [relative_error] [internal norm] [initial epsilon] [epsilon_factor] [max_bids_per_round] [gamma_threshold][log_filename_prefix]. By default power is 1.0, relative error is 0.01, internal norm is l_infinity, initall epsilon is chosen automatically, epsilon factor is 5.0, Jacobi variant is used (max bids per round is maximal), gamma_threshold = 0.0." << std::endl;
        return 1;
    }

    unsigned int dim = std::stoul(argv[3]);

    if (!hera::read_diagram_dipha<double, PairVector>(argv[1], dim, diagramA)) {
        std::exit(1);
    }

    if (!hera::read_diagram_dipha<double, PairVector>(argv[2], dim, diagramB)) {
        std::exit(1);
    }

    params.wasserstein_power = (5 <= argc) ? atof(argv[4]) : 1.0;
    if (params.wasserstein_power < 1.0) {
        std::cerr << "The third argument (wasserstein_degree) was \"" << argv[4] << "\", must be a number >= 1.0. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    if (params.wasserstein_power == 1.0) {
        hera::remove_duplicates<double>(diagramA, diagramB);
    }

    //default relative error:  1%
    params.delta = (6 <= argc) ? atof(argv[5]) : 0.01;
    if ( params.delta <= 0.0) {
        std::cerr << "The 4th argument (relative error) was \"" << argv[5] << "\", must be a number > 0.0. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    // default for internal metric is l_infinity
    params.internal_p = ( 7 <= argc ) ? atof(argv[6]) : hera::get_infinity<double>();
    if (std::isinf(params.internal_p)) {
        params.internal_p = hera::get_infinity<double>();
    }


    if (not hera::is_p_valid_norm<double>(params.internal_p)) {
        std::cerr << "The 6th argument (internal norm) was \"" << argv[6] << "\", must be a number >= 1.0 or inf. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    // if you want to specify initial value for epsilon and the factor
    // for epsilon-scaling
    params.initial_epsilon= ( 8 <= argc ) ? atof(argv[7]) : 0.0 ;

    if (params.initial_epsilon < 0.0) {
        std::cerr << "The 7th argument (initial epsilon) was \"" << argv[7] << "\", must be a non-negative number. Cannot proceed." << std::endl;
        std::exit(1);
    }

    params.epsilon_common_ratio = ( 9 <= argc ) ? atof(argv[8]) : 0.0 ;
    if (params.epsilon_common_ratio <= 1.0 and params.epsilon_common_ratio != 0.0) {
        std::cerr << "The 8th argument (epsilon factor) was \"" << argv[8] << "\", must be a number greater than 1. Cannot proceed." << std::endl;
        std::exit(1);
    }


    params.max_bids_per_round = ( 10 <= argc ) ? atoi(argv[9]) : 0;
    if (params.max_bids_per_round == 0)
        params.max_bids_per_round = std::numeric_limits<decltype(params.max_bids_per_round)>::max();


    params.gamma_threshold = (11 <= argc) ? atof(argv[10]) : 0.0;

    std::string log_filename_prefix = ( 12 <= argc ) ? argv[11] : "";

    params.max_num_phases = 800;

#ifdef LOG_AUCTION
    spdlog::set_level(spdlog::level::info);
#endif

    double res = hera::wasserstein_dist(diagramA, diagramB, params, log_filename_prefix);

    std::cout << std::setprecision(15) << res << std::endl;

    return 0;

}
