.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

Examples and Tutorials
======================

In this folder you can find basic tutorials and examples: you can read through them to understand how giotto-tda works.

Classifying Shapes
------------------

This tutorial is about generating classical surfaces, such as tori and 2-spheres, and study their cohomological properites.
Non-orientable surfaces, such as the Klein bottle, are approximated by a grid and the reciprocal distances between the grid
vertices forms the input of the Vietoris-Rips algorithm.

Lorenz attractor
----------------

This tutorial is about detecting chaotic regimes in a simulation of the `Lorenz attractor. <https://en.wikipedia.org/wiki/Lorenz_system>`_
The main tools of giotto-tda useful for time-series analysis (such as the *Takens Embedding*) are used and explained in the tutorial.
Other feature creation methods, such as the *persistence Landscape* or the *persistence Entropy* are described in the final part of the
tutorial.

Can there be non trivial H_2 in 2-dimensions?
--------------------------------------------

This is a simple riddle that shows how the Vietoris-Rips algorithm may find counterintuitive patters in point-clouds.
The second homology group, H_2, describes and counts voids: for example, the 2-sphere has a non-trivial H_2. Therefore,
we would not expect to find voids in 2-dimensional flat space! On the other hand, it is enough to carefully position 6 points
on the plane to get a nontrivial H_2: check the example out for an empirical proof!
