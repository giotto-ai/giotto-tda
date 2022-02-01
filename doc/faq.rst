
===
FAQ
===

I am a researcher. Can I use ``giotto-tda`` in my project?
----------------------------------------------------------
.. _L2F team: business@l2f.ch

Of course! The `license <https://github.com/giotto-ai/giotto-tda/blob/master/LICENSE>`_ is very permissive.
For more information, please contact the `L2F team`_.

How do I cite ``giotto-tda``?
-----------------------------
We would appreciate citations to the following paper:

   `giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration <https://www.jmlr.org/papers/volume22/20-325/20-325.pdf>`_, Tauzin *et al*, J. Mach. Learn. Res. 22.39 (2021): 1-6.

You can use the following BibTeX entry:

.. code:: RST

    @article{giotto-tda,
      author  = {Guillaume Tauzin and Umberto Lupo and Lewis Tunstall and Julian Burella P\'{e}rez and Matteo Caorsi and Anibal M. Medina-Mardones and Alberto Dassatti and Kathryn Hess},
      title   = {giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration},
      journal = {Journal of Machine Learning Research},
      year    = {2021},
      volume  = {22},
      number  = {39},
      pages   = {1-6},
      url     = {http://jmlr.org/papers/v22/20-325.html}
    }

I cannot install ``giotto-tda``
-------------------------------

We are trying our best to support a variety of widely-used operating systems. Please navigate to
:ref:`Installation <installation>` and review the steps outlined there. Take care of the differences
between a simple user installation and a more involved developer installation from sources.
If you still experience issues, it is possible others also have encountered and reported them.
Please consult the list of `issues <https://github.com/giotto-ai/giotto-tda/issues?q=is%3Aissue>`_,
including the closed ones, and open a new one in case you did not find help.

There are many TDA libraries available. How is ``giotto-tda`` different?
------------------------------------------------------------------------

``giotto-tda`` is oriented towards machine learning (for details, see the :ref:`guiding principles <guiding_principles>`).
This philosophy is in contrast with other reference libraries, like `GUDHI <https://gudhi.inria.fr/doc/latest/index.html>`_,
which provide more low-level functionality at the expense of being less adapted to e.g. batch processing, or of
being tightly integrated with ``scikit-learn``.
