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

#include <iomanip>
#include "bottleneck.h"

// any container of pairs of doubles can be used,
// we use vector in this example.

typedef std::vector<std::pair<double, double>>  PairVector;

// estimate initial guess on sampled diagram?
constexpr bool useSamplingHeur = false;
// if diagrams contain fewer points, don't use heuristic
constexpr int heurThreshold = 30000;

int main(int argc, char* argv[])
{
    if (argc < 3 ) {
        std::cerr << "Usage: " << argv[0] << " file1 file2 [relative_error]. Without relative_error calculate the exact distance." << std::endl;
        return 1;
    }

    PairVector diagramA, diagramB;
    int decPrecision { 0 };
    if (!hera::readDiagramPointSet(argv[1], diagramA, decPrecision)) {
        std::exit(1);
    }

    if (!hera::readDiagramPointSet(argv[2], diagramB, decPrecision)) {
        std::exit(1);
    }

    double res;
    hera::bt::MatchingEdge<double> e;
    if (argc >= 4) {
        // the third parameter is epsilon,
        // return approximate distance (faster)
        double delta =  atof(argv[3]);
        if (delta > 0.0) {
            if (useSamplingHeur && diagramA.size() > heurThreshold && diagramB.size() > heurThreshold) {
                res = hera::bottleneckDistApproxHeur(diagramA, diagramB, delta);
            } else {
                res = hera::bottleneckDistApprox(diagramA, diagramB, delta, e, true);
            }
        } else if (delta == 0.0) {
            res = hera::bottleneckDistExact(diagramA, diagramB, decPrecision);
        } else {
            std::cerr << "The third parameter (relative error) must be positive!" << std::endl;
            std::exit(1);
        }
    } else {
        // only filenames have been supplied, return exact distance
        res = hera::bottleneckDistExact(diagramA, diagramB, decPrecision, e, true);

    }
    std::cout << std::setprecision(15) << res << std::endl;
    //std::cout << "Longest edge " << e.first.get_user_id() << " <-> " << e.second.get_user_id() << std::endl;
    //std::cout << "Longest edge " << e.first << " <-> " << e.second << std::endl;
    // Alternative could be to construct DiagramPointSet
    // using the constructor with iterators.
    // May be useful if the same diagram is used multiple times
    // to avoid copying data from user's container each time.

    //hera::bt::DiagramPointSet dA(diagramA);
    //hera::bt::DiagramPointSet dB(diagramB);
    //double result1 = hera::bt::bottleneckDistExact(dA, dB);
    //std::cout << std::setprecision(15) << result1 << std::endl;

    return 0;
}
