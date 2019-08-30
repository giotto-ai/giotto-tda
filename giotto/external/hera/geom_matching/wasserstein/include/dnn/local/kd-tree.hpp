#include <boost/range/counting_range.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include <boost/range.hpp>

#include <queue>
#include <stack>

#include "../parallel/tbb.h"
#include "def_debug_ws.h"

template<class T>
hera::ws::dnn::KDTree<T>::
KDTree(const Traits& traits, HandleContainer&& handles, double _wassersteinPower):
    traits_(traits), tree_(std::move(handles)), wassersteinPower(_wassersteinPower)
{ assert(wassersteinPower >= 1.0); init(); }

template<class T>
template<class Range>
hera::ws::dnn::KDTree<T>::
KDTree(const Traits& traits, const Range& range, double _wassersteinPower):
    traits_(traits), wassersteinPower(_wassersteinPower)
{
    assert( wassersteinPower >= 1.0);
    init(range);
}

template<class T>
template<class Range>
void
hera::ws::dnn::KDTree<T>::
init(const Range& range)
{
    size_t sz = std::distance(std::begin(range), std::end(range));
    tree_.reserve(sz);
    weights_.resize(sz, 0);
    subtree_weights_.resize(sz, 0);
    for (PointHandle h : range)
        tree_.push_back(h);
    init();
}

template<class T>
void
hera::ws::dnn::KDTree<T>::
init()
{
    if (tree_.empty())
        return;

#if defined(TBB)
    task_group g;
    g.run(OrderTree(tree_.begin(), tree_.end(), 0, traits()));
    g.wait();
#else
    OrderTree(tree_.begin(), tree_.end(), 0, traits()).serial();
#endif

    for (size_t i = 0; i < tree_.size(); ++i)
        indices_[tree_[i]] = i;
}

template<class T>
struct
hera::ws::dnn::KDTree<T>::OrderTree
{
                OrderTree(HCIterator b_, HCIterator e_, size_t i_, const Traits& traits_):
                    b(b_), e(e_), i(i_), traits(traits_)            {}

    void        operator()() const
    {
        if (e - b < 1000)
        {
            serial();
            return;
        }

        HCIterator m = b + (e - b)/2;
        CoordinateComparison cmp(i, traits);
        std::nth_element(b,m,e, cmp);
        size_t next_i = (i + 1) % traits.dimension();

        task_group g;
        if (b < m - 1)  g.run(OrderTree(b,   m, next_i, traits));
        if (e > m + 2)  g.run(OrderTree(m+1, e, next_i, traits));
        g.wait();
    }

    void        serial() const
    {
        std::queue<KDTreeNode> q;
        q.push(KDTreeNode(b,e,i));
        while (!q.empty())
        {
            HCIterator b, e; size_t i;
            std::tie(b,e,i) = q.front();
            q.pop();
            HCIterator m = b + (e - b)/2;

            CoordinateComparison cmp(i, traits);
            std::nth_element(b,m,e, cmp);
            size_t next_i = (i + 1) % traits.dimension();

            // Replace with a size condition instead?
            if (m - b > 1)  q.push(KDTreeNode(b,   m, next_i));
            if (e - m > 2)  q.push(KDTreeNode(m+1, e, next_i));
        }
    }

    HCIterator      b, e;
    size_t          i;
    const Traits&   traits;
};

template<class T>
template<class ResultsFunctor>
void
hera::ws::dnn::KDTree<T>::
search(PointHandle q, ResultsFunctor& rf) const
{
    typedef         typename HandleContainer::const_iterator        HCIterator;
    typedef         std::tuple<HCIterator, HCIterator, size_t>      KDTreeNode;

    if (tree_.empty())
        return;

    DistanceType    D  = std::numeric_limits<DistanceType>::max();

    // TODO: use tbb::scalable_allocator for the queue
    std::queue<KDTreeNode>  nodes;

    nodes.push(KDTreeNode(tree_.begin(), tree_.end(), 0));

    while (!nodes.empty())
    {
        HCIterator b, e; size_t i;
        std::tie(b,e,i) = nodes.front();
        nodes.pop();

        CoordinateComparison cmp(i, traits());
        i = (i + 1) % traits().dimension();

        HCIterator   m    = b + (e - b)/2;

        DistanceType dist = (wassersteinPower == 1.0) ?  traits().distance(q, *m) + weights_[m - tree_.begin()] : std::pow(traits().distance(q, *m), wassersteinPower) + weights_[m - tree_.begin()];


        D = rf(*m, dist);

        // we are really searching w.r.t L_\infty ball; could prune better with an L_2 ball
        Coordinate diff = cmp.diff(q, *m);     // diff returns signed distance

        DistanceType diffToWasserPower = (wassersteinPower == 1.0) ? diff : ((diff > 0 ? 1.0 : -1.0) * std::pow(fabs(diff), wassersteinPower));

        size_t       lm   = m + 1 + (e - (m+1))/2 - tree_.begin();
        if (e > m + 1 && diffToWasserPower - subtree_weights_[lm] >= -D) {
            nodes.push(KDTreeNode(m+1, e, i));
        }

        size_t       rm   = b + (m - b) / 2 - tree_.begin();
        if (b < m && diffToWasserPower + subtree_weights_[rm] <= D) {
            nodes.push(KDTreeNode(b,   m, i));
        }
    }
}

template<class T>
void
hera::ws::dnn::KDTree<T>::
adjust_weights(DistanceType delta)
{
    for(auto& w : weights_)
        w -= delta;

    for(auto& sw : subtree_weights_)
        sw -= delta;
}


template<class T>
void
hera::ws::dnn::KDTree<T>::
change_weight(PointHandle p, DistanceType w)
{
    size_t idx = indices_[p];

    if ( weights_[idx] == w ) {
        return;
    }

    bool weight_increases = ( weights_[idx] < w );
    weights_[idx] = w;

    typedef     std::tuple<HCIterator, HCIterator>      KDTreeNode;

    // find the path down the tree to this node
    // not an ideal strategy, but // it's not clear how to move up from the node in general
    std::stack<KDTreeNode>  s;
    s.push(KDTreeNode(tree_.begin(),tree_.end()));

    do
    {
        HCIterator b,e;
        std::tie(b,e) = s.top();

        size_t im = b + (e - b)/2 - tree_.begin();

        if (idx == im)
            break;
        else if (idx < im)
            s.push(KDTreeNode(b, tree_.begin() + im));
        else    // idx > im
            s.push(KDTreeNode(tree_.begin() + im + 1, e));
    } while(1);

    // update subtree_weights_ on the path to the root
    DistanceType min_w = w;
    while (!s.empty())
    {
        HCIterator b,e;
        std::tie(b,e) = s.top();
        HCIterator m  = b + (e - b)/2;
        size_t     im = m - tree_.begin();
        s.pop();


        // left and right children
        if (b < m)
        {
            size_t lm = b + (m - b)/2 - tree_.begin();
            if (subtree_weights_[lm] < min_w)
                min_w = subtree_weights_[lm];
        }

        if (e > m + 1)
        {
            size_t rm = m + 1 + (e - (m+1))/2 - tree_.begin();
            if (subtree_weights_[rm] < min_w)
                min_w = subtree_weights_[rm];
        }

        if (weights_[im] < min_w) {
            min_w = weights_[im];
        }

        if (weight_increases) {

            if (subtree_weights_[im] < min_w )   // increase weight
                subtree_weights_[im] = min_w;
            else
                break;

        } else {

            if (subtree_weights_[im] > min_w )   // decrease weight
                subtree_weights_[im] = min_w;
            else
                break;

        }
    }
}

template<class T>
typename hera::ws::dnn::KDTree<T>::HandleDistance
hera::ws::dnn::KDTree<T>::
find(PointHandle q) const
{
    hera::ws::dnn::NNRecord<HandleDistance> nn;
    search(q, nn);
    return nn.result;
}

template<class T>
typename hera::ws::dnn::KDTree<T>::Result
hera::ws::dnn::KDTree<T>::
findR(PointHandle q, DistanceType r) const
{
    hera::ws::dnn::rNNRecord<HandleDistance> rnn(r);
    search(q, rnn);
    std::sort(rnn.result.begin(), rnn.result.end());
    return rnn.result;
}

template<class T>
typename hera::ws::dnn::KDTree<T>::Result
hera::ws::dnn::KDTree<T>::
findK(PointHandle q, size_t k) const
{
    hera::ws::dnn::kNNRecord<HandleDistance> knn(k);
    search(q, knn);
    std::sort(knn.result.begin(), knn.result.end());
    return knn.result;
}


template<class T>
struct hera::ws::dnn::KDTree<T>::CoordinateComparison
{
                CoordinateComparison(size_t i, const Traits& traits):
                    i_(i), traits_(traits)                              {}

    bool        operator()(PointHandle p1, PointHandle p2) const        { return coordinate(p1) < coordinate(p2); }
    Coordinate  diff(PointHandle p1, PointHandle p2) const              { return coordinate(p1) - coordinate(p2); }

    Coordinate  coordinate(PointHandle p) const                         { return traits_.coordinate(p, i_); }
    size_t      axis() const                                            { return i_; }

    private:
        size_t          i_;
        const Traits&   traits_;
};

template<class T>
void
hera::ws::dnn::KDTree<T>::
printWeights(void)
{
#ifndef FOR_R_TDA
    std::cout << "weights_:" << std::endl;
    for(const auto ph : indices_) {
        std::cout << "idx = " << ph.second << ": (" << (ph.first)->at(0) << ", " << (ph.first)->at(1) << ") weight = " << weights_[ph.second] << std::endl;
    }
    std::cout << "subtree_weights_:" << std::endl;
    for(size_t idx = 0; idx < subtree_weights_.size(); ++idx) {
        std::cout << idx << " : " << subtree_weights_[idx] << std::endl;
    }
#endif
}


