.. -*- mode: rst -*-

|Azure|_ 

.. |Azure| image:: https://dev.azure.com/matteocaorsi/matteocao/_apis/build/status/matteocao.giotto-learn?branchName=master
.. _Azure: https://dev.azure.com/matteocaorsi/matteocao/


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
