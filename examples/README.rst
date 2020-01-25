.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

Examples, tutorials and plotting utilities
==========================================

In this folder you can find basic tutorials and examples to get started quickly with giotto-tda. Additionally, ``plotting.py`` contains utilities for plotting the outputs of several computations you can perform with giotto-tda.

Classifying Shapes
------------------

This tutorial is about generating classical surfaces, such as tori and 2-spheres, and study their cohomological properites.
Non-orientable surfaces, such as the Klein bottle, are approximated by a grid and the reciprocal distances between the grid
vertices forms the input of the Vietoris-Rips algorithm.

Lorenz attractor
----------------

This tutorial is about detecting chaotic regimes in a simulation of the `Lorenz attractor <https://en.wikipedia.org/wiki/Lorenz_system>`_. The main tools of giotto-tda useful for time-series analysis (such as the *Takens embedding*) are used and explained in the tutorial. Other feature creation methods, such as the *persistence landscape* or the *persistence entropy*, are described in the final part of the
tutorial.

Mapper quickstart
-----------------

The Mapper algorithm was introduced in v0.1.5, and allows you to visualize complex, high-dimensional data in a simple way as a graph to reveal structural insights. This tutorial covers some of the main functionalities of the ``gtda.mapper`` module.

Can there be non trivial H\ :sub:`2` in 2 dimensions?
-----------------------------------------------------

This is a simple riddle that shows how the Vietoris-Rips algorithm may find counterintuitive patters in point-clouds.
The second homology group, H\ :sub:`2`, describes and counts voids: for example, the 2-sphere has a non-trivial H\ :sub:`2`. Therefore, we would not expect to find voids in 2-dimensional flat space! On the other hand, it is enough to carefully position 6 points on the plane to get a nontrivial H\ :sub:`2`: check the example out for an empirical proof!
