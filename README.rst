.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

|Azure|_ |Azure-cov|_ |Azure-test|_ |binder|_

.. |Azure| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-learn?branchName=master
.. _Azure: https://dev.azure.com/maintainers/Giotto/_build/latest?definitionId=2&branchName=master

.. |Azure-cov| image:: https://img.shields.io/badge/Coverage-93%25-passed
.. _Azure-cov: https://dev.azure.com/maintainers/Giotto/_build/results?buildId=6&view=codecoverage-tab

.. |Azure-test| image:: https://img.shields.io/badge/Testing-Passed-brightgreen
.. _Azure-test: https://dev.azure.com/maintainers/Giotto/_build/results?buildId=6&view=ms.vss-test-web.build-test-results-tab

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/giotto-ai/giotto-learn/master?filepath=examples


giotto-learn
============


giotto-learn is a high performance topological machine learning toolbox in Python built on top of
scikit-learn and is distributed under the GNU AGPLv3 license. It is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects.

Website: https://giotto.ai


Project genesis
---------------

giotto-learn is the result of a collaborative effort between `L2F SA
<https://www.l2f.ch/>`_, the `Laboratory for Topology and Neuroscience
<https://www.epfl.ch/labs/hessbellwald-lab/>`_ at EPFL, and the `Institute of Reconfigurable & Embedded Digital Systems (REDS)
<https://heig-vd.ch/en/research/reds>`_ of HEIG-VD.

Installation
------------

Dependencies
~~~~~~~~~~~~

The latest stable version of giotto-learn requires:

- Python (>= 3.5)
- NumPy (>= 1.17.0)
- SciPy (>= 0.17.0)
- joblib (>= 0.11)
- scikit-learn (>= 0.21.3)

Additionally, developer or pre-release versions require:

- scikit-learn (>= 0.22.0)
- python-igraph (>= 0.7.1.post6)
- matplotlib (>= 3.1.2)
- plotly (>= 4.4.1)
- ipywidgets

To run the examples, jupyter is required.

User installation
~~~~~~~~~~~~~~~~~

The simplest way to install giotto-learn is using ``pip``   ::

    pip install -U giotto-learn

Pre-release, experimental builds containing recently added features and/or
bug fixes can be installed by running   ::

    pip install -U giotto-learn-nightly

The main difference between ``giotto-learn-nightly`` and the developer
installation (see below) is that the former is shipped with pre-compiled wheels
(similarly to the stable release) and hence does not require any C++ dependencies.

Documentation
-------------

- HTML documentation (stable release): https://docs.giotto.ai

Contributing
------------

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to giotto-learn, please see the `CONTRIBUTING.rst
<https://github.com/giotto-ai/giotto-learn/blob/master/CONTRIBUTING.rst>`_ file.

Developer installation
~~~~~~~~~~~~~~~~~~~~~~~

To simplify the installation of the C++ dependencies, we recommend using
`Conda <https://www.anaconda.com/distribution/>`_ to install giotto-learn from source.

Installing both the PyPI release and source of giotto-learn in the same
environment is not recommended since it is known to cause conflicts with the C++ bindings.

C++ dependencies:
'''''''''''''''''

-  C++14 compatible compiler
-  CMake >= 3.9
-  Boost >= 1.56

The CMake and Boost dependencies can be installed using Conda as follows:

.. code-block:: bash

    conda install -c anaconda cmake
    conda install -c anaconda boost

Source code
'''''''''''

You can check the latest sources with the command::

    git clone https://github.com/giotto-ai/giotto-learn.git


To install:
'''''''''''

.. code-block:: bash

   cd giotto-learn
   pip install -e ".[tests, doc]"

From there any change in the library files will be immediately available on your machine.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory::

    pytest giotto


Changelog
---------

See the `RELEASE.rst <https://github.com/giotto-ai/giotto-learn/blob/master/RELEASE.rst>`__ file
for a history of notable changes to giotto-learn.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/giotto-ai/giotto-learn
- Download releases: https://pypi.org/project/giotto-learn/
- Issue tracker: https://github.com/giotto-ai/giotto-learn/issues

Community
---------

giotto-ai Slack workspace: https://slack.giotto.ai/

Contacts
--------

maintainers@giotto.ai
