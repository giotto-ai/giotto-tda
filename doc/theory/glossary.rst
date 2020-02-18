Symbols
=======

+---------------------+---+------------------------------------------+
| :math:`\Bbbk`       | : | An arbitrary field.                      |
+---------------------+---+------------------------------------------+
| :math:`\mathbb R^d` | : | The vector space of :math:`d`-tuples of  |
|                     |   | real numbers.                            |
+---------------------+---+------------------------------------------+
| :math:`\mathbb N`   | : | The counting numbers                     |
|                     |   | :math:`0,1,2, \dots` as a subset of      |
|                     |   | :math:`\mathbb R`.                       |
+---------------------+---+------------------------------------------+
| :math:`\Delta`      | : | The multiset                             |
|                     |   | :math:`\{(s,s)\,;\ s \in \mathbb R\}`    |
|                     |   | with multiplicity                        |
|                     |   | :math:`(s,s) \mapsto +\infty`.           |
+---------------------+---+------------------------------------------+

Homology
========

Cubical complex
---------------

An **elementary interval** :math:`I_a` is a subset of :math:`\mathbb{R}`
of the form :math:`[a, a+1]` or :math:`[a,a] = \{a\}` for some
:math:`a \in \mathbb{R}`. These two types are called respectively
**non-degenerate** and **degenerate**. To a non-degenerate elementary
interval we assign two degenerate elementary intervals

.. math:: d^+ I_a = [a+1, a+1] \qquad \text{and} \qquad d^- I_a = [a, a].

An **elementary cube** is a subset of the form

.. math:: I_{a_1} \times \cdots \times I_{a_N} \subset \mathbb{R}^N

where each :math:`I_{a_i}` is an elementary interval. We refer to the
total number of its non-degenerate factors
:math:`I_{a_{k_1}}, \dots, I_{a_{k_n}}` as its **dimension** and,
assuming

.. math:: a_{k_1} < \cdots < a_{k_{n,}}

we define for :math:`i = 1, \dots, n` the following two elementary cubes

.. math:: d_i^\pm I^N = I_{a_1} \times \cdots \times d^\pm I_{a_{k_i}} \times \cdots \times I_{a_{N.}}

A **cubical complex** is a finite set of elementary cubes of
:math:`\mathbb{R}^N`, and a **subcomplex** of :math:`X` is a cubical
complex whose elementary cubes are also in :math:`X`. We denote the set
of :math:`n`-dimensional cubes as :math:`X_n`.

Reference:
~~~~~~~~~~

(Kaczynski, Mischaikow, and Mrozek 2004)

Simplicial complex
------------------

A set :math:`\{v_0, \dots, v_n\} \subset \mathbb{R}^N` is said to be
**geometrically independent** if the vectors
:math:`\{v_0-v_1, \dots, v_0-v_n\}` are linearly independent. In this
case, we refer to their convex closure as a **simplex**, explicitly

.. math:: = \left\{ \sum c_i (v_0 - v_i)\ \big|\ c_1+\dots+c_n = 1,\ c_i \geq 0 \right\}

and to :math:`n` as its **dimension**. The **:math:`i`-th face** of
:math:`[v_0, \dots, v_n]` is defined for :math:`i=0, \dots, n` by

.. math:: d_i[v_0, \dots, v_n] = [v_0, \dots, \widehat{v}_i, \dots, v_n]

where :math:`\widehat{v}_i` denotes the absence of :math:`v_i` from the
set.

A **simplicial complex** :math:`X` is a finite union of simplices in
:math:`\mathbb{R}^N` satisfying that every face of a simplex in
:math:`X` is in :math:`X` and that the non-empty intersection of two
simplices in :math:`X` is a face of each. Every simplicial complex
defines an abstract simplicial complex.

.. _simplicial complex:

Abstract simplicial complex:
----------------------------

An *abstract simplicial complex* is a pair of sets :math:`(V, X)` with
the elements of :math:`X` being subsets of :math:`V` such that:

#. for every :math:`v` in :math:`V`, the singleton :math:`\{v\}` is in
   :math:`X` and

#. if :math:`x` is in :math:`X` and :math:`y` is a subset of :math:`x`,
   then :math:`y` is in :math:`X`.

We abuse notation and denote the pair :math:`(V, X)` simply by
:math:`X`.

The elements of :math:`X` are called *simplices* and the *dimension* of
a simplex :math:`x` is defined by :math:`|x| = \# x - 1` where
:math:`\# x` denotes the cardinality of :math:`x`. Simplices of
dimension :math:`d` are called :math:`d`-simplices. We abuse terminology
and refer to the elements of :math:`V` and to their associated
:math:`0`-simplices both as *vertices*.

The *:math:`k`-skeleton* :math:`X_k` of a simplicial complex :math:`X`
is the subcomplex containing all simplices of dimension at most
:math:`k`. A simplicial complex is said to be *:math:`d`-dimensional* if
:math:`d` is the smallest integer satisfying :math:`X = X_d`.

A *simplicial map* between simplicial complexes is a function between
their vertices such that the image of any simplex via the induced map is
a simplex.

A simplicial complex :math:`X` is a *subcomplex* of a simplicial complex
:math:`Y` if every simplex of :math:`X` is a simplex of :math:`Y`.

Given a finite abstract simplicial complex :math:`X = (V, X)` we can
choose a bijection from :math:`V` to a geometrically independent subset
of :math:`\mathbb R^N` and associate a simplicial complex to :math:`X`
called its *geometric realization*.

Ordered simplicial complex:
---------------------------

An *ordered simplicial complex* is an abstract simplicial complex where
the set of vertices is equipped with a partial order such that the
restriction of this partial order to any simplex is a total order. We
denote an :math:`n`-simplex using its ordered vertices by
:math:`[v_0, \dots, v_n]`.

A *simplicial map* between ordered simplicial complexes is a simplicial
map :math:`f` between their underlying simplicial complexes preserving
the order, i.e., :math:`v \leq w` implies :math:`f(v) \leq f(w)`.

Directed simplicial complex:
----------------------------

A *directed simplicial complex* is a pair of sets :math:`(V, X)` with
the elements of :math:`X` being tuples of elements of :math:`V`, i.e.,
elements in :math:`\bigcup_{n\geq1} V^{\times n}` such that:

#. for every :math:`v` in :math:`V`, the tuple :math:`v` is in :math:`X`
   and

#. if :math:`x` is in :math:`X` and :math:`y` is a subtuple of
   :math:`x`, then :math:`y` is in :math:`X`.

With appropriate modifications the same terminology and notation
introduced for ordered simplicial complex applies to directed simplicial
complex.

Clique or flag complexes:
-------------------------

Let :math:`G` be a :math:`1`-dimensional simplicial complex, abstract or
otherwise. The complex :math:`\langle G \rangle` has the same set of
vertices as :math:`G` and :math:`\{v_0, \dots, v_n\}` is a simplex in
:math:`\langle G \rangle` if an only if :math:`\{v_i, v_j\} \in G` for
each pair of vertices :math:`v_i, v_j`.

Let :math:`G` be a :math:`1`-dimensional directed simplicial complex.
The directed simplicial complex :math:`\langle G \rangle` has the same
set of vertices as :math:`G` and :math:`(v_0, \dots, v_n)` is a simplex
in :math:`\langle G \rangle` if an only if :math:`(v_i, v_j) \in G` for
each pair of vertices :math:`v_i, v_j` with :math:`i < j`.

A (directed) simplicial complex :math:`X` is a *clique complex* a.k.a.
*flag complex* if :math:`X = \langle X_1 \rangle` where :math:`X_1` is
the :math:`1`-skeleton of :math:`X`.

Chain complex:
--------------

A *chain complex* of is a pair :math:`(C_*, \partial)` where

.. math:: C_* = \bigoplus_{n \in \mathbb Z} C_n \quad \mathrm{and} \quad \partial = \bigoplus_{n \in \mathbb Z} \partial_n

with :math:`C_n` a :math:`\Bbbk`-vector space and
:math:`\partial_n : C_{n+1} \to C_n` is a :math:`\Bbbk`-linear map such
that :math:`\partial_{n+1} \partial_n = 0`. We refer to :math:`\partial`
as the *boundary map* of the chain complex.

The elements of :math:`C` are called *chains* and if :math:`c \in C_n`
we say its *degree* is :math:`n` or simply that it is an
:math:`n`-chain. Elements in the kernel of :math:`\partial` are called
*cycles*, and elements in the image of :math:`\partial` are called
*boundaries*. Notice that every boundary is a cycle. This fact is
central to the definition of homology.

A *chain map* is a :math:`\Bbbk`-linear map :math:`f : C \to C'` between
chain complexes such that :math:`f(C_n) \subseteq C'_n` and
:math:`\partial f = f \partial`.

Given a chain complex :math:`(C_*, \partial)`, its linear dual
:math:`C^*` is also a chain complex with
:math:`C^{-n} = \mathrm{Hom_\Bbbk}(C_n, \Bbbk)` and boundary map
:math:`\delta` defined by :math:`\delta(\alpha)(c) = \alpha(\partial c)`
for any :math:`\alpha \in C^*` and :math:`c \in C_*`.

Homology and cohomology:
------------------------

Let :math:`(C_*, \partial)` be a chain complex. Its *:math:`n`-th
homology group* is the quotient of the subspace of :math:`n`-cycles by
the subspace of :math:`n`-boundaries, that is,
:math:`H_n(C_*) = \mathrm{ker}(\partial_n)/ \mathrm{im}(\partial_{n+1})`.
The *homology* of :math:`(C, \partial)` is defined by
:math:`H_*(C) = \bigoplus_{n \in \mathbb Z} H_n(C)`.

When the chain complex under consideration is the linear dual of a chain
complex we sometimes refer to its homology as the *cohomology* of the
predual complex and write :math:`H^n` for :math:`H_{-n}`.

A chain map :math:`f : C \to C'` induces a map between the associated
homologies.

Simplicial chains and simplicial homology:
------------------------------------------

Let :math:`X` be an ordered or directed simplicial complex. Define its
*simplicial chain complex with :math:`\Bbbk`-coefficients*
:math:`C_*(X; \Bbbk)` by

.. math:: C_n(X; \Bbbk) = \Bbbk\{X_n\} \qquad \partial_n(x) = \sum_{i=0}^{n} (-1)^i d_ix

and its *homology and cohomology with :math:`\Bbbk`-coefficients* as the
homology and cohomology of this chain complex. We use the notation
:math:`H_*(X; \Bbbk)` and :math:`H^*(X; \Bbbk)` for these.

A simplicial map induces a chain map between the associated simplicial
chain complexes and, therefore, between the associated simplicial
(co)homologies.

Cubical chains and cubical homology:
------------------------------------

Let :math:`X` be a cubical complex. Define its *cubical chain complex
with :math:`\Bbbk`-coefficients* :math:`C_*(X; \Bbbk)` by

.. math:: C_n(X; \Bbbk) = \Bbbk\{X_n\} \qquad \partial_n x = \sum_{i = 1}^{n} (-1)^{i-1}(d^+_i x - d^-_i x)

where :math:`x = I_1 \times \cdots \times I_N` and :math:`s(i)` is the
dimension of :math:`I_1 \times \cdots \times I_i`. Its *homology and
cohomology with :math:`\Bbbk`-coefficients* is the homology and
cohomology of this chain complex. We use the notation
:math:`H_*(X; \Bbbk)` and :math:`H^*(X; \Bbbk)` for these.

Filtered complex:
-----------------

A *filtered complex* is a collection of simplicial or of cubical
complexes :math:`\{X(n)\}_{n \in \mathbb N}` such that :math:`X(n)` is a
subcomplex of :math:`X(n+1)` for each :math:`n \geq 0`.

Cellwise filtration:
--------------------

A filtered complex such that :math:`X(n+1)` contains exactly one more
simplex or elementary cube than :math:`X(n)`.

The data of a simplexwise filtration is equivalent to a complex
:math:`X` together with a total order :math:`\leq` on its simplices or
elementary cubes such that for each :math:`y \in X` the set
:math:`\{x \in X\ :\ x \leq y\}` is a subcomplex of :math:`X`.

Persistence module:
-------------------

A *persistence module* is a collection containing a :math:`\Bbbk`-vector
spaces :math:`V(s)` for each real number :math:`s` together with
:math:`\Bbbk`-linear maps :math:`f_{st} : V(s) \to V(t)`, referred to as
*structure maps*, for each pair :math:`s \leq t`, satisfying naturality,
i.e., if :math:`r \leq s \leq t`, then
:math:`f_{rt} = f_{st} \circ f_{rs}` and tameness, i.e., all but
finitely many structure maps are isomorphisms.

A *morphism of persistence modules* :math:`F : V \to W` is a collection
of linear maps :math:`F(s) : V(s) \to W(s)` such that
:math:`F(t) \circ f_{st} = f_{st} \circ F(s)` for each par of reals
:math:`s \leq t`. We say that :math:`F` is an *isomorphisms* if each
:math:`F(s)` is.

Persistent simplicial (co)homology:
-----------------------------------

Let :math:`\{X(s)\}_{s \in \mathbb R}` be a collection of ordered or
directed simplicial complexes together with simplicial maps
:math:`f_{st} : X(s) \to X(t)` for each pair :math:`s \leq t`, such that
if :math:`r \leq s \leq t`, then :math:`f_{rt} = f_{st} \circ f_{rs}`.
Its *persistent simplicial homology with :math:`\Bbbk`-coefficients* is
the persistence module

.. math:: H_*(X(s); \Bbbk)

with structure maps
:math:`H_*(f_{st}) : H_*(X(s); \Bbbk) \to H_*(X(t); \Bbbk)` induced form
the maps :math:`f_{st.}` In general, the collection constructed this way
needs not satisfy the tameness condition of a persistence module, but we
restrict attention to the cases where it does.

*Persistence simplicial cohomology with :math:`\Bbbk`-coefficients* is
defined analogously.

Vietoris-Rips complex and Vietoris-Rips homology:
-------------------------------------------------

The *Vietoris-Rips complex* of a finite metric space :math:`(X, d)` is
given by the following construction: let :math:`s \geq 0`, define the
simplicial complex :math:`VR_X(s)` to have vertices the set :math:`X`
and declare a subset :math:`\{x_0, \dots, x_n\}` of distinct points in
:math:`X` to be a simplex if :math:`d(x_i, x_j) \leq s` for all
:math:`x_i, x_j`, explicitly

.. math:: VR_X(s) = \big\{ [v_0,\dots,v_n]\ |\ d(v_i,v_j) \leq s \text{ for all } i,j = 0,\dots n \big\}.

We equipped this collection of complexes with the inclusion maps
:math:`VR_X(s) \to VR_X(t)` for each :math:`s \leq t` and define the
Vietoris-Rips homology of :math:`(X, d)` to be the persistent simplicial
homology of this collection.

Multiset:
---------

A *multiset* is a pair :math:`(S, \phi)` where :math:`S` is a set and
:math:`\phi : S \to \mathbb N  \cup \{+\infty\}` is a function attaining
positive values. For :math:`s \in S` we refer to :math:`\phi(s)` as its
*multiplicity*. The *union* of two multisets
:math:`(S_1, \phi_1), (S_2, \phi_2)` is the multiset
:math:`(S_1 \cup S_2, \phi_1 \cup \phi_2)` with

.. math::

   (\phi_1 \cup \phi_2)(s) = 
       \begin{cases}
       \phi_1(s) & s \in S_1, s \not\in S_2 \\
       \phi_2(s) & s \in S_2, s \not\in S_1 \\
       \phi_1(s) + \phi_2(s) & s \in S_1, s \in S_2. \\
       \end{cases}

Persistence diagram:
--------------------

A *persistence diagram* is a multiset of points in
:math:`\mathbb R \times \mathbb{R} \cup \{+\infty\}`.

Given a persistence module its associated persistence diagram is
determined by the following condition: for each pair :math:`s,t` the
number counted with multiplicity of points :math:`(b,d)` in the
multiset, satisfying :math:`b \leq s \leq t < d` is equal to the rank of
:math:`f_{st.}`

A well known result establishes that there exists an isomorphism between
two persistence module if and only if their persistence diagrams are
equal.

Wasserstein and bottleneck distance:
------------------------------------

The *:math:`p`-Wasserstein distance* between two persistence diagrams
:math:`D_1` and :math:`D_2` is the infimum over all bijections
:math:`\gamma: D_1 \cup \Delta \to D_2 \cup \Delta` of

.. math:: \sum_{x \in D_1 \cup \Delta} \Big(||x - \gamma(x)||_\infty^p \Big)^{1/p}

where :math:`||-||_\infty` is defined for :math:`(x,y) \in \mathbb R^2`
by :math:`\max\{|x|, |y|\}`.

The limit :math:`p \to \infty` defines the *bottleneck distance*. More
explicitly, it is the infimum over the same set of bijections of the
value

.. math:: \sup_{x \in D_1 \cup \Delta} ||x - \gamma(x)||_{\infty.}

.. _reference-1:

Reference:
~~~~~~~~~~

(Kerber, Morozov, and Nigmetov 2017)

Persistence landscape:
----------------------

A *persistence landscape* is a continuous function

.. math:: \lambda : \mathbb N \times  \mathbb R \to \mathbb R \cup \{+\infty\}

and the function :math:`\lambda_k(s) = \lambda(k,s)` is refered to as
the *:math:`k`-layer of the persistence diagram*.

Let :math:`\{(b_i, d_i)\}_{i \in I}` be the persistence diagram of a
persistence module. Its *associated persistence landscape*
:math:`\lambda` is defined by letting :math:`\lambda_k(t)` be
:math:`k`-th largest value of

.. math:: \min_{i \in I}\{t-b_i, d_i-t\}_+

where :math:`c_+` denotes :math:`max(c,0)`.

Intuitively, we can describe the graph of this persistence landscape by
first joining each of the points in the multiset to the diagonal via a
horizontal as well as a vertical line, then rotating the figure 45
degrees clockwise, and rescaling by :math:`1/\sqrt{2}`.

References:
~~~~~~~~~~~

(Bubenik 2015)

Persistence landscape norm:
---------------------------

Given a function
:math:`f : \mathbb R \to \overline{\mathbb R} = [-\infty, +\infty]`
define

.. math:: ||f||_p = \left( \int_{\mathbb R} f^p(x)\, dx \right)^{1/p}

whenever the right hand side exists and is finite.

The *persistence landscape :math:`p`-norm* of a persistence landscape
:math:`\lambda : \mathbb N \times \mathbb R \to \overline{\mathbb R}` is
defined to be

.. math:: ||\lambda||_p = \left( \sum_{i \in \mathbb N} ||\lambda_i||^p_p \right)^{1/p}

whenever the right hand side exists and is finite.

.. _references-1:

References:
~~~~~~~~~~~

(Stein and Shakarchi 2011; Bubenik 2015)

Amplitude
---------

Given a function assigning a real number to a pair of persistence
diagrams. We define the *amplitude* of a persistence diagram :math:`D`
to be the value assigned to the pair :math:`(D \cup \Delta, \Delta)`.
Important examples of such functions are: Wasserstein and bottleneck
distances and landscape distance.

Persistence entropy:
--------------------

Intuitively, this is a measure of the entropy of the points in a
persistence diagram. Precisely, let :math:`D = \{(b_i, d_i)\}_{i \in I}`
be a persistence diagram with each :math:`d_i < +\infty`. The
*persistence entropy* of :math:`D` is defined by

.. math:: E(D) = - \sum_{i \in I} p_i \log(p_i)

where

.. math:: p_i = \frac{(d_i - b_i)}{L_D} \qquad \text{and} \qquad L_D = \sum_{i \in I} (d_i - b_i) .

.. _references-2:

References:
~~~~~~~~~~~

(Rucco et al. 2016)

Betti curve:
------------

Let :math:`D` be a persistence diagram. Its *Betti curve* is the
function :math:`\beta_D : \mathbb R \to \mathbb N` whose value on
:math:`s \in \mathbb R` is the number, counted with multiplicity, of
points :math:`(b_i,d_i)` in :math:`D` such that :math:`b_i \leq s <d_i`.

The name is inspired from the case when the persistence diagram comes
from persistent homology.

Metric space:
-------------

A pair :math:`(X, d)` where :math:`X` is a set and :math:`d` is a
function

.. math:: d : X \times X \to \mathbb R

attaining non-negative values is called a *metric space* if

.. math:: d(x,y) = 0\ \Leftrightarrow\  x = y

.. math:: d(x,y) = d(y,x)

.. math:: d(x,z) \leq d(x,y) + d(y, z)

In this case, the function :math:`d` is refer to as the *metric* and the
value :math:`d(x,y)` is called the *distance* between :math:`x` and
:math:`y`.

Euclidean distance and norm:
----------------------------

The set :math:`\mathbb R^n` defines a metric space with euclidean
distance

.. math:: d(x,y) = \sqrt{(x_1-y_1)^2 + \cdots + (x_n-y_n)^2}.

The norm :math:`||x||` of a vector :math:`x` is defined as its distance
to the :math:`0` vector.

Finite metric spaces and point clouds:
--------------------------------------

A *finite metric space* is a finite set together with a metric. A
*distance matrix* associated to a finite metric space is obtained by
choosing a total order on the finite set and setting the
:math:`(i,j)`-entry to be equal to the distance between the :math:`i`-th
and :math:`j`-th elements.

A *point cloud* is a finite subset of :math:`\mathbb{R}^n` (for some
:math:`n`) together with the metric induced from the eucliden distance.

Time series
===========

.. _time-series-1:

Time series:
------------

A *time series* is a sequence :math:`\{y_i\}_{i = 0}^n` of real numbers.

A common construction of a times series :math:`\{x_i\}_{i = 0}^n` is
given by choosing :math:`x_0` arbitrarily as well as a step parameter
:math:`h` and setting

.. math:: x_i = x_0 + h\cdot i.

Another usual construction is as follows: given a time series
:math:`\{x_i\}_{i = 0}^n \subseteq U` and a function

.. math:: f :  U \subseteq \mathbb R \to \mathbb R

we obtain a new time series :math:`\{f(x_i)\}_{i = 0.}^n`

Generalizing the previous construction we can define a time series from
a function

.. math:: \varphi : U  \times M \to M, \qquad U \subseteq \mathbb R, \qquad M \subseteq \mathbb R^d

using a function :math:`f : M \to \mathbb R` as follows: let
:math:`\{t_i\}_{i=0}^n` be a time series taking values in :math:`U`,
then

.. math:: \{f(\varphi(t_i, m))\}_{i=0}^n.

for an arbitrarily chosen :math:`m \in M`.

Takens embedding:
-----------------

Let :math:`M \subset \mathbb R^d` be a compact manifold of dimension
:math:`n`. Let

.. math:: \varphi : \mathbb R  \times M \to M

and

.. math:: f : M \to \mathbb R

be generic smooth functions. Then, for any :math:`\tau > 0` the map

.. math:: M \to \mathbb R^{2n+1}

defined by

.. math:: x \mapsto\big( f(x), f(x_1), f(x_2), \dots, f(x_{2n}) \big)

where

.. math:: x_i = \varphi(i \cdot \tau, x)

is an injective map with full rank.

.. _reference-2:

*Reference*:
~~~~~~~~~~~~

(Takens 1981)

Manifold:
---------

Intuitively, a manifold of dimension :math:`n` is a space locally
equivalent to :math:`\mathbb R^n`. Formally, a subset :math:`M` of
:math:`\mathbb R^d` is an :math:`n`-dimensional manifold if for each
:math:`x \in M` there exists an open ball
:math:`B(x) = \{ y \in M\,;\ d(x,y) < \epsilon\}` and a smooth function
with smooth inverse

.. math:: \phi_x : B(x) \to \{v \in \mathbb R^n\,;\ ||v||<1\}.

.. _references-3:

*References*:
~~~~~~~~~~~~~

(Milnor and Weaver 1997; Guillemin and Pollack 2010)

Compact subset:
---------------

A subset :math:`K` of a metric space :math:`(X,d)` is said to be
*bounded* if there exist a real number :math:`D` such that for each pair
of elements in :math:`K` the distance between them is less than
:math:`D`. It is said to be *complete* if for any :math:`x \in X` it is
the case that :math:`x \in K` if for any :math:`\epsilon > 0` the
intersection between :math:`K` and :math:`\{y \,;\ d(x,y) < \epsilon \}`
is not empty. It is said to be *compact* if it is both bounded and
complete.

Bibliography
============

.. container:: references hanging-indent
   :name: refs

   .. container::
      :name: ref-bubenik2015statistical

      Bubenik, Peter. 2015. “Statistical Topological Data Analysis Using
      Persistence Landscapes.” *The Journal of Machine Learning
      Research* 16 (1): 77–102.

   .. container::
      :name: ref-guillemin2010differential

      Guillemin, Victor, and Alan Pollack. 2010. *Differential
      Topology*. Vol. 370. American Mathematical Soc.

   .. container::
      :name: ref-mischaikow04computational

      Kaczynski, Tomasz, Konstantin Mischaikow, and Marian Mrozek. 2004.
      *Computational Homology*. Vol. 157. Applied Mathematical Sciences.
      Springer-Verlag, New York. https://doi.org/10.1007/b97315.

   .. container::
      :name: ref-kerber2017geometry

      Kerber, Michael, Dmitriy Morozov, and Arnur Nigmetov. 2017.
      “Geometry Helps to Compare Persistence Diagrams.” *Journal of
      Experimental Algorithmics (JEA)* 22: 1–4.

   .. container::
      :name: ref-milnor1997topology

      Milnor, John Willard, and David W Weaver. 1997. *Topology from the
      Differentiable Viewpoint*. Princeton university press.

   .. container::
      :name: ref-rucco2016characterisation

      Rucco, Matteo, Filippo Castiglione, Emanuela Merelli, and Marco
      Pettini. 2016. “Characterisation of the Idiotypic Immune Network
      Through Persistent Entropy.” In *Proceedings of Eccs 2014*,
      117–28. Springer.

   .. container::
      :name: ref-stein2011functional

      Stein, Elias M, and Rami Shakarchi. 2011. *Functional Analysis:
      Introduction to Further Topics in Analysis*. Vol. 4. Princeton
      University Press.

   .. container::
      :name: ref-takens1981detecting

      Takens, Floris. 1981. “Detecting Strange Attractors in
      Turbulence.” In *Dynamical Systems and Turbulence, Warwick 1980*,
      366–81. Springer.
