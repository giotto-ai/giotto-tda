.. -*- mode: rst -*-

|Azure|_ |Azure-cov|_ |Azure-test|_

.. |Azure| image:: https://dev.azure.com/giotto-learn/giotto-learn/_apis/build/status/giotto-learn.giotto-learn?branchName=master
.. _Azure: https://dev.azure.com/giotto-learn/giotto-learn/

.. |Azure-cov| image:: https://coveralls.io/repos/neovim/neovim/badge.svg?branch=master
.. _Azure-cov: https://dev.azure.com/giotto-learn/giotto-learn/_build/results?buildId=364&view=codecoverage-tab

.. |Azure-test| image:: https://travis-ci.org/scikit-learn/scikit-learn.svg?branch=master
.. _Azure-test: https://dev.azure.com/giotto-learn/giotto-learn/_build/results?buildId=364&view=ms.vss-test-web.build-test-results-tab


giotto-learn
============

C++ dependencies:
-----------------
-  C++14 compatible compiler
-  CMake >= 3.0
-  Boost >= 1.56

To install:
-----------

.. code-block:: bash

   git clone https://github.com/giotto-learn/giotto-learn.git
   cd giotto-learn
   git submodule update --init --recursive
   pip install .


To use:
-------

.. code-block:: python

   import giotto as gt
