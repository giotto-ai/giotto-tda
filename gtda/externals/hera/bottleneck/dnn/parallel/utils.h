#ifndef HERA_BT_PARALLEL_UTILS_H
#define HERA_BT_PARALLEL_UTILS_H

#include "../utils.h"

namespace hera
{
namespace bt
{
namespace dnn
{
    // Assumes rng is synchronized across ranks
    template<class DataVector, class RNGType, class SwapFunctor>
    void shuffle(mpi::communicator& world, DataVector& data, RNGType& rng, const SwapFunctor& swap, DataVector empty = DataVector());

    template<class DataVector, class RNGType>
    void shuffle(mpi::communicator& world, DataVector& data, RNGType& rng)
    {
        typedef     decltype(data[0])           T;
        shuffle(world, data, rng, [](T& x, T& y) { std::swap(x,y); });
    }
}
}
}

template<class DataVector, class RNGType, class SwapFunctor>
void
hera::bt::dnn::shuffle(mpi::communicator& world, DataVector& data, RNGType& rng, const SwapFunctor& swap, DataVector empty)
{
    // This is not a perfect shuffle: it dishes out data in chunks of 1/size.
    // (It can be interpreted as generating a bistochastic matrix by taking the
    // sum of size random permutation matrices.) Hopefully, it works for our purposes.

    typedef     typename RNGType::result_type       RNGResult;

    int size = world.size();
    int rank = world.rank();

    // Generate local seeds
    boost::uniform_int<RNGResult> uniform;
    RNGResult seed;
    for (size_t i = 0; i < size; ++i)
    {
        RNGResult v = uniform(rng);
        if (i == rank)
            seed = v;
    }
    RNGType local_rng(seed);

    // Shuffle local data
    hera::bt::dnn::random_shuffle(data.begin(), data.end(), local_rng, swap);

    // Decide how much of our data goes to i-th processor
    std::vector<size_t>     out_counts(size);
    std::vector<int>        ranks(boost::counting_iterator<int>(0),
                                  boost::counting_iterator<int>(size));
    for (size_t i = 0; i < size; ++i)
    {
        hera::bt::dnn::random_shuffle(ranks.begin(), ranks.end(), rng);
        ++out_counts[ranks[rank]];
    }

    // Fill the outgoing array
    size_t total = 0;
    std::vector< DataVector > outgoing(size, empty);
    for (size_t i = 0; i < size; ++i)
    {
        size_t count = data.size()*out_counts[i]/size;
        if (total + count > data.size())
            count = data.size() - total;

        outgoing[i].reserve(count);
        for (size_t j = total; j < total + count; ++j)
            outgoing[i].push_back(data[j]);

        total += count;
    }

    boost::uniform_int<size_t> uniform_outgoing(0,size-1);  // in range [0,size-1]
    while(total < data.size())                              // send leftover to random processes
    {
        outgoing[uniform_outgoing(local_rng)].push_back(data[total]);
        ++total;
    }
    data.clear();

    // Exchange the data
    std::vector< DataVector > incoming(size, empty);
    mpi::all_to_all(world, outgoing, incoming);
    outgoing.clear();

    // Assemble our data
    for(const DataVector& vec : incoming)
        for (size_t i = 0; i < vec.size(); ++i)
            data.push_back(vec[i]);
    hera::bt::dnn::random_shuffle(data.begin(), data.end(), local_rng, swap);
    // XXX: the final shuffle is irrelevant for our purposes. But it's also cheap.
}

#endif
