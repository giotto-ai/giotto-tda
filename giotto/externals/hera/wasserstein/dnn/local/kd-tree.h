#ifndef HERA_WS_DNN_LOCAL_KD_TREE_H
#define HERA_WS_DNN_LOCAL_KD_TREE_H

#include "../utils.h"
#include "search-functors.h"

#include <unordered_map>

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/range/value_type.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

namespace hera
{
namespace ws
{
namespace dnn
{
    // Weighted KDTree
    // Traits_ provides Coordinate, DistanceType, PointType, dimension(), distance(p1,p2), coordinate(p,i)
    template< class Traits_ >
    class KDTree
    {
        public:
            typedef         Traits_                                         Traits;
            typedef         dnn::HandleDistance<KDTree>                     HandleDistance;

            typedef         typename Traits::PointType                      Point;
            typedef         typename Traits::PointHandle                    PointHandle;
            typedef         typename Traits::Coordinate                     Coordinate;
            typedef         typename Traits::DistanceType                   DistanceType;
            typedef         std::vector<PointHandle>                        HandleContainer;
            typedef         std::vector<HandleDistance>                     HDContainer;   // TODO: use tbb::scalable_allocator
            typedef         HDContainer                                     Result;
            typedef         std::vector<DistanceType>                       DistanceContainer;
            typedef         std::unordered_map<PointHandle, size_t>         HandleMap;

            BOOST_STATIC_ASSERT_MSG(has_coordinates<Traits, PointHandle, int>::value, "KDTree requires coordinates");

        public:
                            KDTree(const Traits& traits):
                                traits_(traits)                             {}

                            KDTree(const Traits& traits, HandleContainer&& handles, double _wassersteinPower = 1.0);

            template<class Range>
                            KDTree(const Traits& traits, const Range& range, double _wassersteinPower = 1.0);

            template<class Range>
            void            init(const Range& range);

            DistanceType    weight(PointHandle p) { return weights_[indices_[p]]; }
            void            change_weight(PointHandle p, DistanceType w);
            void            adjust_weights(DistanceType delta);             // subtract delta from all weights

            HandleDistance  find(PointHandle q) const;
            Result          findR(PointHandle q, DistanceType r) const;     // all neighbors within r
            Result          findK(PointHandle q, size_t k) const;           // k nearest neighbors

            HandleDistance  find(const Point& q) const                      { return find(traits().handle(q)); }
            Result          findR(const Point& q, DistanceType r) const     { return findR(traits().handle(q), r); }
            Result          findK(const Point& q, size_t k) const           { return findK(traits().handle(q), k); }

            template<class ResultsFunctor>
            void            search(PointHandle q, ResultsFunctor& rf) const;

            const Traits&   traits() const                                  { return traits_; }

            void printWeights(void);

        private:
            void            init();

            typedef     typename HandleContainer::iterator                  HCIterator;
            typedef     std::tuple<HCIterator, HCIterator, size_t>          KDTreeNode;

            struct CoordinateComparison;
            struct OrderTree;

        private:
            Traits              traits_;
            HandleContainer     tree_;
            DistanceContainer   weights_;               // point weight
            DistanceContainer   subtree_weights_;       // min weight in the subtree
            HandleMap           indices_;
            double wassersteinPower;
    };
} // dnn
} // ws
} // hera

#include "kd-tree.hpp"

#endif
