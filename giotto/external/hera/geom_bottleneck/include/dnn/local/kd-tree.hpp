#include <boost/range/counting_range.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include <boost/range.hpp>

#include <queue>

#include "../parallel/tbb.h"

template<class T>
hera::bt::dnn::KDTree<T>::KDTree(const Traits& traits, HandleContainer&& handles):
    traits_(traits),
    tree_(std::move(handles)),
    delete_flags_(handles.size(), static_cast<char>(0) ),
    subtree_n_elems(handles.size(), static_cast<size_t>(0)),
    num_points_(handles.size())
{
    init();
}

template<class T>
template<class Range>
hera::bt::dnn::KDTree<T>::KDTree(const Traits& traits, const Range& range):
    traits_(traits)
{
    init(range);
}

template<class T>
template<class Range>
void hera::bt::dnn::KDTree<T>::init(const Range& range)
{
    size_t sz = std::distance(std::begin(range), std::end(range));
    subtree_n_elems = std::vector<int>(sz, 0);
    delete_flags_ = std::vector<char>(sz, 0);
    num_points_ = sz;
    tree_.reserve(sz);
    for (PointHandle h : range)
        tree_.push_back(h);
    parents_.resize(sz, -1);
    init();
}

template<class T>
void hera::bt::dnn::KDTree<T>::init()
{
    if (tree_.empty())
        return;

#if defined(TBB)
    task_group g;
    g.run(OrderTree(this, tree_.begin(), tree_.end(), -1, 0, traits()));
    g.wait();
#else
    OrderTree(this, tree_.begin(), tree_.end(), -1, 0, traits()).serial();
#endif

    for (size_t i = 0; i < tree_.size(); ++i)
        indices_[tree_[i]] = i;
    init_n_elems();
}

template<class T>
struct
hera::bt::dnn::KDTree<T>::OrderTree
{
                OrderTree(KDTree* tree_, HCIterator b_, HCIterator e_, ssize_t p_, size_t i_, const Traits& traits_):
                    tree(tree_), b(b_), e(e_), p(p_), i(i_), traits(traits_)            {}

    void        operator()() const
    {
        if (e - b < 1000)
        {
            serial();
            return;
        }

        HCIterator m  = b + (e - b)/2;
        ssize_t    im = m - tree->tree_.begin();
        tree->parents_[im]  = p;

        CoordinateComparison cmp(i, traits);
        std::nth_element(b,m,e, cmp);
        size_t next_i = (i + 1) % traits.dimension();

        task_group g;
        if (b < m - 1)  g.run(OrderTree(tree, b,   m, im, next_i, traits));
        if (e > m + 2)  g.run(OrderTree(tree, m+1, e, im, next_i, traits));
        g.wait();
    }

    void        serial() const
    {
        std::queue<KDTreeNode> q;
        q.push(KDTreeNode(b,e,p,i));
        while (!q.empty())
        {
            HCIterator b, e; ssize_t p; size_t i;
            std::tie(b,e,p,i) = q.front();
            q.pop();
            HCIterator m  = b + (e - b)/2;
            ssize_t    im = m - tree->tree_.begin();
            tree->parents_[im]  = p;

            CoordinateComparison cmp(i, traits);
            std::nth_element(b,m,e, cmp);
            size_t next_i = (i + 1) % traits.dimension();

            // Replace with a size condition instead?
            if (m - b > 1)
                q.push(KDTreeNode(b,   m, im, next_i));
            else if (b < m)
                tree->parents_[im - 1] = im;
            if (e - m > 2)
                q.push(KDTreeNode(m+1, e, im, next_i));
            else if (e > m + 1)
                tree->parents_[im + 1] = im;
        }
    }

    KDTree*         tree;
    HCIterator      b, e;
    ssize_t         p;
    size_t          i;
    const Traits&   traits;
};

template<class T>
void hera::bt::dnn::KDTree<T>::update_n_elems(ssize_t idx, const int delta)
// add delta to the number of points in node idx and update subtree_n_elems
// for all parents of the node idx
{
    //std::cout << "subtree_n_elems.size = " << subtree_n_elems.size() << std::endl;
    // update the node itself
    while (idx != -1)
    {
        //std::cout << idx << std::endl;
        subtree_n_elems[idx] += delta;
        idx = parents_[idx];
    }
}

template<class T>
void hera::bt::dnn::KDTree<T>::increase_n_elems(const ssize_t idx)
{
    update_n_elems(idx, static_cast<ssize_t>(1));
}

template<class T>
void hera::bt::dnn::KDTree<T>::decrease_n_elems(const ssize_t idx)
{
    update_n_elems(idx, static_cast<ssize_t>(-1));
}

template<class T>
void hera::bt::dnn::KDTree<T>::init_n_elems()
{
    for(size_t idx = 0; idx < tree_.size(); ++idx) {
        increase_n_elems(idx);
    }
}


template<class T>
template<class ResultsFunctor>
void hera::bt::dnn::KDTree<T>::search(PointHandle q, ResultsFunctor& rf) const
{
    typedef         typename HandleContainer::const_iterator        HCIterator;
    typedef         std::tuple<HCIterator, HCIterator, size_t>      KDTreeNode;

    if (tree_.empty())
        return;

    DistanceType    D  = std::numeric_limits<DistanceType>::infinity();

    // TODO: use tbb::scalable_allocator for the queue
    std::queue<KDTreeNode>  nodes;

    nodes.push(KDTreeNode(tree_.begin(), tree_.end(), 0));

    //std::cout << "started kdtree::search" << std::endl;

    while (!nodes.empty())
    {
        HCIterator b, e; size_t i;
        std::tie(b,e,i) = nodes.front();
        nodes.pop();

        CoordinateComparison cmp(i, traits());
        i = (i + 1) % traits().dimension();

        HCIterator   m    = b + (e - b)/2;
        size_t m_idx = m - tree_.begin();
        // ignore deleted points
        if ( delete_flags_[m_idx] == 0 ) {
            DistanceType dist = traits().distance(q, *m);
            // + weights_[m - tree_.begin()];
            //std::cout << "Supplied to functor: m : ";
            //std::cout << "(" << (*(*m))[0] << ", " << (*(*m))[1] << ")";
            //std::cout << " and q : ";
            //std::cout << "(" << (*q)[0] << ", " << (*q)[1] << ")" << std::endl;
            //std::cout << "dist^q + weight = " << dist << std::endl;
            //std::cout << "weight = " << weights_[m - tree_.begin()] << std::endl;
            //std::cout << "dist = " << traits().distance(q, *m) << std::endl;
            //std::cout << "dist^q = " << pow(traits().distance(q, *m), wassersteinPower) << std::endl;

            D = rf(*m, dist);
        }
        // we are really searching w.r.t L_\infty ball; could prune better with an L_2 ball
        Coordinate diff = cmp.diff(q, *m);     // diff returns signed distance
        DistanceType diffToWasserPower = (diff > 0 ? 1.0 : -1.0) * fabs(diff);

        size_t       lm   = m + 1 + (e - (m+1))/2 - tree_.begin();
        if ( e > m + 1 and subtree_n_elems[lm] > 0 ) {
            if (e > m + 1 && diffToWasserPower  >= -D) {
                nodes.push(KDTreeNode(m+1, e, i));
            }
        }

        size_t       rm   = b + (m - b) / 2 - tree_.begin();
        if ( subtree_n_elems[rm] > 0 ) {
            if (b < m && diffToWasserPower  <= D) {
                nodes.push(KDTreeNode(b,   m, i));
            }
        }
    }
    //std::cout << "exited kdtree::search" << std::endl;
}

template<class T>
typename hera::bt::dnn::KDTree<T>::HandleDistance hera::bt::dnn::KDTree<T>::find(PointHandle q) const
{
    hera::bt::dnn::NNRecord<HandleDistance> nn;
    search(q, nn);
    return nn.result;
}

template<class T>
typename hera::bt::dnn::KDTree<T>::Result hera::bt::dnn::KDTree<T>::findR(PointHandle q, DistanceType r) const
{
    hera::bt::dnn::rNNRecord<HandleDistance> rnn(r);
    search(q, rnn);
    //std::sort(rnn.result.begin(), rnn.result.end());
    return rnn.result;
}

template<class T>
typename hera::bt::dnn::KDTree<T>::Result hera::bt::dnn::KDTree<T>::findFirstR(PointHandle q, DistanceType r) const
{
    hera::bt::dnn::firstrNNRecord<HandleDistance> rnn(r);
    search(q, rnn);
    return rnn.result;
}

template<class T>
typename hera::bt::dnn::KDTree<T>::Result hera::bt::dnn::KDTree<T>::findK(PointHandle q, size_t k) const
{
    hera::bt::dnn::kNNRecord<HandleDistance> knn(k);
    search(q, knn);
    // do we need this???
    std::sort(knn.result.begin(), knn.result.end());
    return knn.result;
}

template<class T>
struct hera::bt::dnn::KDTree<T>::CoordinateComparison
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
void hera::bt::dnn::KDTree<T>::delete_point(const size_t idx)
{
    // prevent double deletion
    assert(delete_flags_[idx] == 0);
    delete_flags_[idx] = 1;
    decrease_n_elems(idx);
    --num_points_;
}

template<class T>
void hera::bt::dnn::KDTree<T>::delete_point(PointHandle p)
{
    delete_point(indices_[p]);
}

