
|Azure|_ |Azure-cov|_ |Azure-test|_ |binder|_

.. |Azure| image:: https://dev.azure.com/giotto-learn/giotto-learn/_apis/build/status/giotto-learn.giotto-learn?branchName=master
.. _Azure: https://dev.azure.com/giotto-learn/giotto-learn/

.. |Azure-cov| image:: https://img.shields.io/badge/Coverage-93%25-passed
.. _Azure-cov: https://dev.azure.com/giotto-learn/giotto-learn/_build/results?buildId=342&view=codecoverage-tab

.. |Azure-test| image:: https://img.shields.io/badge/Testing-Passed-brightgreen
.. _Azure-test: https://dev.azure.com/giotto-learn/giotto-learn/_build/results?buildId=342&view=ms.vss-test-web.build-test-results-tab

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/giotto-learn/giotto-learn/master?filepath=examples

giotto-learn
============


giotto-learn is a Python module for topological data analysis in machine learning pipelines built on top of
scikit-learn and is distributed under the Apache 2.0 license.


Project Governance
------------------

The project was started jointly by `Learn To Forecast - L2F <http://www.l2f.ch>`_, `EPFL Laboratory for topology and neuroscience <https://www.epfl.ch/labs/hessbellwald-lab/>`_ and the `Reconfigurable and Embedded Digital Systems at heig-vd <http://reds.heig-vd.ch/en>`_. 

The code is under active development and is maintained and developped by members of those three institutions. See the `GOVERNANCE.rst <https://github.com/giotto-learn/giotto-learn/blob/master/GOVERNANCE.rst>`_ file for a list of the Giotto team members.

Website: http://ww.giotto.ai


Installation
------------

Dependencies
~~~~~~~~~~~~

giotto-learn requires:

- Python (>= 3.5)
- scikit-learn (>= 0.21.3)
- NumPy (>= 1.11.0)
- SciPy (>= 0.17.0)
- joblib (>= 0.11)

For running the examples jupyter, matplotlib and plotly are required.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy,
the easiest way to install scikit-learn is using ``pip``   ::

    pip install -U giotto-learn

Development
-----------

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective.

Developper installation
~~~~~~~~~~~~~~~~~~~~~~~

C++ dependencies:
'''''''''''''''''

-  C++14 compatible compiler
-  CMake >= 3.9
-  Boost >= 1.56

To install:
'''''''''''

.. code-block:: bash

   git clone https://github.com/giotto-learn/giotto-learn.git
   cd giotto-learn
   pip install -e .


Changelog
---------

See the `changelog <https://github.com/giotto-learn/giotto-learn/blob/master/RELEASE.rst>`__
for a history of notable changes to giotto-learn.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/giotto-learn/giotto-learn
- Download releases: https://pypi.org/project/giotto-learn/
- Issue tracker: https://github.com/giotto-learn/giotto-learn/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/giotto-learn/giotto-learn.git

Contributing
~~~~~~~~~~~~

To learn more about making a contribution to scikit-learn, please see the
`CONTRIBUTING.rst
<https://github.com/giotto-learn/giotto-learn/blob/master/CONTRIBUTING.rst>`_ file.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory::

    pytest giotto


Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before opening a Pull Request, have a look at the
full Contributing page to make sure your code complies
with our guidelines: http://scikit-learn.org/stable/developers/index.html


Documentation
~~~~~~~~~~~~~

- HTML documentation (stable release): http://www.giotto.ai/docs/

Contacts:
---------

maintainers@giotto.ai
