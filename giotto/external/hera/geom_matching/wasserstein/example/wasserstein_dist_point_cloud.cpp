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

#include "wasserstein_pure_geom.hpp"

int main(int argc, char* argv[])
{

    //{
    //int n_points = 3;
    //int dim = 3;
    //using Traits = hera::ws::dnn::DynamicPointTraits<double>;
    //hera::ws::dnn::DynamicPointTraits<double> traits(dim);
    //hera::ws::dnn::DynamicPointVector<double> dgm_a = traits.container(n_points);
    //hera::ws::dnn::DynamicPointVector<double> dgm_b = traits.container(n_points);

    //dgm_a[0][0] = 0.0;
    //dgm_a[0][1] = 0.0;
    //dgm_a[0][2] = 0.0;

    //dgm_a[1][0] = 1.0;
    //dgm_a[1][1] = 0.0;
    //dgm_a[1][2] = 0.0;

    //dgm_a[2][0] = 0.0;
    //dgm_a[2][1] = 1.0;
    //dgm_a[2][2] = 1.0;

    //dgm_b[0][0] = 0.0;
    //dgm_b[0][1] = 0.1;
    //dgm_b[0][2] = 0.1;

    //dgm_b[1][0] = 1.1;
    //dgm_b[1][1] = 0.0;
    //dgm_b[1][2] = 0.0;

    //dgm_b[2][0] = 0.0;
    //dgm_b[2][1] = 1.1;
    //dgm_b[2][2] = 0.9;


    //hera::AuctionParams<double> params;
    //params.dim = dim;


    //std::cout << hera::ws::wasserstein_cost(dgm_a, dgm_b, params) << std::endl;

    //return 0;
    //}


    using Real = double;
    using PointVector = hera::ws::dnn::DynamicPointVector<Real>;
    PointVector set_A, set_B;

    hera::AuctionParams<Real> params;

    if (argc < 3 ) {
        std::cerr << "Usage: " << argv[0] << " file1 file2 [wasserstein_degree] [relative_error] [internal norm] [initial epsilon] [epsilon_factor] [max_bids_per_round] [gamma_threshold][log_filename_prefix]. By default power is 1.0, relative error is 0.01, internal norm is l_infinity, initall epsilon is chosen automatically, epsilon factor is 5.0, Jacobi variant is used (max bids per round is maximal), gamma_threshold = 0.0." << std::endl;
        return 1;
    }

    int dimension_A, dimension_B;

    if (!hera::read_point_cloud<Real>(argv[1], set_A, dimension_A)) {
        std::exit(1);
    }

    if (!hera::read_point_cloud(argv[2], set_B, dimension_B)) {
        std::exit(1);
    }

    if (dimension_A != dimension_B) {
        std::cerr << "Dimension mismatch: " << dimension_A << " != " << dimension_B << std::endl;
        std::exit(1);
    }

    params.dim = dimension_A;

    params.wasserstein_power = (4 <= argc) ? atof(argv[3]) : 1.0;
    if (params.wasserstein_power < 1.0) {
        std::cerr << "The third argument (wasserstein_degree) was \"" << argv[3] << "\", must be a number >= 1.0. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    //if (params.wasserstein_power == 1.0) {
    //    hera::remove_duplicates<double?(set_A, set_B);
    //}

    //default relative error:  1%
    params.delta = (5 <= argc) ? atof(argv[4]) : 0.01;
    if ( params.delta <= 0.0) {
        std::cerr << "The 4th argument (relative error) was \"" << argv[4] << "\", must be a number > 0.0. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    // default for internal metric is l_infinity
    params.internal_p = ( 6 <= argc ) ? atof(argv[5]) : hera::get_infinity<Real>();
    if (std::isinf(params.internal_p)) {
        params.internal_p = hera::get_infinity<Real>();
    }


    if (not hera::is_p_valid_norm<Real>(params.internal_p)) {
        std::cerr << "The 5th argument (internal norm) was \"" << argv[5] << "\", must be a number >= 1.0 or inf. Cannot proceed. " << std::endl;
        std::exit(1);
    }

    // if you want to specify initial value for epsilon and the factor
    // for epsilon-scaling
    params.initial_epsilon= ( 7 <= argc ) ? atof(argv[6]) : 0.0 ;

    if (params.initial_epsilon < 0.0) {
        std::cerr << "The 6th argument (initial epsilon) was \"" << argv[6] << "\", must be a non-negative number. Cannot proceed." << std::endl;
        std::exit(1);
    }

    params.epsilon_common_ratio = ( 8 <= argc ) ? atof(argv[7]) : 0.0 ;
    if (params.epsilon_common_ratio <= 1.0 and params.epsilon_common_ratio != 0.0) {
        std::cerr << "The 7th argument (epsilon factor) was \"" << argv[7] << "\", must be a number greater than 1. Cannot proceed." << std::endl;
        std::exit(1);
    }


    params.max_bids_per_round = ( 9 <= argc ) ? atoi(argv[8]) : 0;
    if (params.max_bids_per_round == 0)
        params.max_bids_per_round = std::numeric_limits<decltype(params.max_bids_per_round)>::max();


    params.gamma_threshold = (10 <= argc) ? atof(argv[9]) : 0.0;

    std::string log_filename_prefix = ( 11 <= argc ) ? argv[10] : "";

    params.max_num_phases = 800;

#ifdef LOG_AUCTION
    spdlog::set_level(spdlog::level::critical);
#endif

    Real res = hera::ws::wasserstein_dist(set_A, set_B, params);

    std::cout << std::setprecision(15) << res << std::endl;

    return 0;
}
