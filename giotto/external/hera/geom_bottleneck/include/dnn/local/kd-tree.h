#ifndef HERA_BT_DNN_LOCAL_KD_TREE_H
#define HERA_BT_DNN_LOCAL_KD_TREE_H

#include "../utils.h"
#include "search-functors.h"

#include <unordered_map>
#include <stack>

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/range/value_type.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

namespace hera {
namespace bt {
namespace dnn
{
    // Weighted KDTree
    // Traits_ provides Coordinate, DistanceType, PointType, dimension(), distance(p1,p2), coordinate(p,i)
    template< class Traits_ >
    class KDTree
    {
        public:
            typedef         Traits_                                         Traits;
            typedef         hera::bt::dnn::HandleDistance<KDTree>                     HandleDistance;

            typedef         typename Traits::PointType                      Point;
            typedef         typename Traits::PointHandle                    PointHandle;
            typedef         typename Traits::Coordinate                     Coordinate;
            typedef         typename Traits::DistanceType                   DistanceType;
            typedef         std::vector<PointHandle>                        HandleContainer;
            typedef         std::vector<HandleDistance>                     HDContainer;   // TODO: use tbb::scalable_allocator
            typedef         HDContainer                                     Result;
            typedef         std::vector<DistanceType>                       DistanceContainer;
            typedef         std::unordered_map<PointHandle, size_t>         HandleMap;
        //private:
            typedef     typename HandleContainer::iterator                  HCIterator;
            typedef     std::tuple<HCIterator, HCIterator, size_t, ssize_t>     KDTreeNode;
            typedef     std::tuple<HCIterator, HCIterator>          KDTreeNodeNoCut;

            //BOOST_STATIC_ASSERT_MSG(has_coordinates<Traits, PointHandle, int>::value, "KDTree requires coordinates");

        public:
                            KDTree(const Traits& traits):
                                traits_(traits)                             {}

                            KDTree(const Traits& traits, HandleContainer&& handles);

            template<class Range>
                            KDTree(const Traits& traits, const Range& range);

            template<class Range>
            void            init(const Range& range);

            HandleDistance  find(PointHandle q) const;
            Result          findR(PointHandle q, DistanceType r) const;     // all neighbors within r
            Result          findFirstR(PointHandle q, DistanceType r) const;     // first neighbor within r
            Result          findK(PointHandle q, size_t k) const;           // k nearest neighbors

            HandleDistance  find(const Point& q) const                      { return find(traits().handle(q)); }
            Result          findR(const Point& q, DistanceType r) const     { return findR(traits().handle(q), r); }
            Result          findFirstR(const Point& q, DistanceType r) const     { return findFirstR(traits().handle(q), r); }
            Result          findK(const Point& q, size_t k) const           { return findK(traits().handle(q), k); }



            template<class ResultsFunctor>
            void            search(PointHandle q, ResultsFunctor& rf) const;

            const Traits&   traits() const                                  { return traits_; }

            void get_path_to_root(const size_t idx, std::stack<KDTreeNodeNoCut>& s);
            // to support deletion
            void            init_n_elems();
            void            delete_point(const size_t idx);
            void            delete_point(PointHandle p);
            void            update_n_elems(const ssize_t idx, const int delta);
            void            increase_n_elems(const ssize_t idx);
            void            decrease_n_elems(const ssize_t idx);
            size_t          get_num_points() const { return num_points_; }
        //private:
            void            init();


            struct CoordinateComparison;
            struct OrderTree;

        //private:
            Traits              traits_;
            HandleContainer     tree_;
            std::vector<char>   delete_flags_;
            std::vector<int>    subtree_n_elems;
            HandleMap           indices_;
            std::vector<ssize_t> parents_;

            size_t              num_points_;
    };
} // dnn
} // bt
} // hera
#include "kd-tree.hpp"

#endif
