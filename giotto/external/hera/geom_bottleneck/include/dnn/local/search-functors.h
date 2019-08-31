#ifndef HERA_BT_DNN_LOCAL_SEARCH_FUNCTORS_H
#define HERA_BT_DNN_LOCAL_SEARCH_FUNCTORS_H

#include <boost/range/algorithm/heap_algorithm.hpp>

namespace hera
{
namespace bt
{
namespace dnn
{

template<class NN>
struct HandleDistance
{
    typedef             typename NN::PointHandle                                    PointHandle;
    typedef             typename NN::DistanceType                                   DistanceType;
    typedef             typename NN::HDContainer                                    HDContainer;

                        HandleDistance()                                            {}
                        HandleDistance(PointHandle pp, DistanceType dd):
                            p(pp), d(dd)                                            {}
    bool                operator<(const HandleDistance& other) const                { return d < other.d; }

    PointHandle         p;
    DistanceType        d;
};

template<class HandleDistance>
struct NNRecord
{
    typedef         typename HandleDistance::PointHandle                            PointHandle;
    typedef         typename HandleDistance::DistanceType                           DistanceType;

                    NNRecord()                                                      { result.d = std::numeric_limits<DistanceType>::infinity(); }
    DistanceType    operator()(PointHandle p, DistanceType d)                       { if (d < result.d) { result.p = p; result.d = d; } return result.d; }
    HandleDistance  result;
};

template<class HandleDistance>
struct rNNRecord
{
    typedef         typename HandleDistance::PointHandle                            PointHandle;
    typedef         typename HandleDistance::DistanceType                           DistanceType;
    typedef         typename HandleDistance::HDContainer                            HDContainer;

                    rNNRecord(DistanceType r_): r(r_)                               {}
    DistanceType    operator()(PointHandle p, DistanceType d)
    {
        if (d <= r)
            result.push_back(HandleDistance(p,d));
        return r;
    }

    DistanceType    r;
    HDContainer     result;
};

template<class HandleDistance>
struct firstrNNRecord
{
    typedef         typename HandleDistance::PointHandle                            PointHandle;
    typedef         typename HandleDistance::DistanceType                           DistanceType;
    typedef         typename HandleDistance::HDContainer                            HDContainer;

    firstrNNRecord(DistanceType r_): r(r_)                               {}

    DistanceType    operator()(PointHandle p, DistanceType d)
    {
        if (d <= r) {
            result.push_back(HandleDistance(p,d));
            return -100000000.0;
        } else {
            return r;
        }
    }

    DistanceType    r;
    HDContainer     result;
};


template<class HandleDistance>
struct kNNRecord
{
    typedef         typename HandleDistance::PointHandle                            PointHandle;
    typedef         typename HandleDistance::DistanceType                           DistanceType;
    typedef         typename HandleDistance::HDContainer                            HDContainer;

                    kNNRecord(unsigned k_): k(k_)                                   {}
    DistanceType    operator()(PointHandle p, DistanceType d)
    {
        if (result.size() < k)
        {
            result.push_back(HandleDistance(p,d));
            boost::push_heap(result);
            if (result.size() < k)
                return std::numeric_limits<DistanceType>::infinity();
        } else if (d < result[0].d)
        {
            boost::pop_heap(result);
            result.back() = HandleDistance(p,d);
            boost::push_heap(result);
        }
        if ( result.size() > 1 ) {
            assert( result[0].d >= result[1].d );
        }
        return result[0].d;
    }

    unsigned        k;
    HDContainer     result;
};

} // dnn
} // bt
} // hera

#endif // HERA_BT_DNN_LOCAL_SEARCH_FUNCTORS_H
