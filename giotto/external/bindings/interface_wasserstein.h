#include "../hera/geom_matching/wasserstein/include/wasserstein.h"

using namespace std;
using PD = vector<pair<double,double> >;

double wasserstein_dist(PD & diag1, PD & diag2, double p = 1.0, double delta = 0.01){
    hera::AuctionParams<double> params; params.delta = delta; params.wasserstein_power = p; params.internal_p = hera::get_infinity<double>();
    string log_filename_prefix = "";
    return hera::wasserstein_dist(diag1, diag2, params, log_filename_prefix);
}
