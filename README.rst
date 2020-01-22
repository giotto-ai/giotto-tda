.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

|Azure|_ |Azure-cov|_ |Azure-test|_ |binder|_

.. |Azure| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-tda?branchName=master
.. _Azure: https://dev.azure.com/maintainers/Giotto/_build/latest?definitionId=6&branchName=master

.. |Azure-cov| image:: https://img.shields.io/badge/Coverage-93%25-passed
.. _Azure-cov: https://dev.azure.com/maintainers/Giotto/_build/results?buildId=6&view=codecoverage-tab

.. |Azure-test| image:: https://img.shields.io/badge/Testing-Passed-brightgreen
.. _Azure-test: https://dev.azure.com/maintainers/Giotto/_build/results?buildId=6&view=ms.vss-test-web.build-test-results-tab

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/giotto-ai/giotto-tda/master?filepath=examples


giotto-tda
==========


giotto-tda is a high performance topological machine learning toolbox in Python built on top of
scikit-learn and is distributed under the GNU AGPLv3 license. It is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects.

Website: https://giotto.ai


Project genesis
---------------

giotto-tda is the result of a collaborative effort between `L2F SA
<https://www.l2f.ch/>`_, the `Laboratory for Topology and Neuroscience
<https://www.epfl.ch/labs/hessbellwald-lab/>`_ at EPFL, and the `Institute of Reconfigurable & Embedded Digital Systems (REDS)
<https://heig-vd.ch/en/research/reds>`_ of HEIG-VD.

Installation
------------

Dependencies
~~~~~~~~~~~~

The latest stable version of giotto-tda requires:

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

The simplest way to install giotto-tda is using ``pip``   ::

    pip install -U giotto-tda

Note: the above may fail on old versions of ``pip``. We recommend upgrading ``pip``
to a recent version.

Pre-release, experimental builds containing recently added features, and/or
bug fixes can be installed by running   ::

    pip install -U giotto-tda-nightly

The main difference between ``giotto-tda-nightly`` and the developer
installation (see below) is that the former is shipped with pre-compiled wheels
(similarly to the stable release) and hence does not require any C++ dependencies.

Documentation
-------------

- HTML documentation (stable release): https://docs.giotto.ai

Contributing
------------

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to giotto-tda, please see the `CONTRIBUTING.rst
<https://github.com/giotto-ai/giotto-tda/blob/master/CONTRIBUTING.rst>`_ file.

Developer installation
~~~~~~~~~~~~~~~~~~~~~~~

Installing both the PyPI release and source of giotto-tda in the same environment is not recommended since it is
known to cause conflicts with the C++ bindings.

C++ dependencies:
'''''''''''''''''

-  C++14 compatible compiler
-  CMake >= 3.9
-  Boost >= 1.56

Please refer to your system's instructions and to the `CMake <https://cmake.org/>`_ and
`Boost <https://www.boost.org/>`_ websites for definitive guidance on how to install these dependencies. The
instructions below are unofficial, please follow them at your own risk.

- Most Linux systems should come with a suitable compiler pre-installed. For the other two dependencies, you may
  consider running

.. code-block:: bash

    sudo apt-get install cmake
    sudo apt-get install boost

- On macOS, you may consider using ``brew`` (https://brew.sh/) to install the dependencies as follows:

.. code-block:: bash

    brew install gcc
    brew install cmake
    brew install boost

- On Windows, you will likely need to have `Visual Studio <https://visualstudio.microsoft.com/>`_ installed. At present,
  it appears to be important to have a recent version of the VS C++ compiler. One way to check whether this is the case
  is as follows: 1) open the VS Installer GUI; 2) under the "Installed" tab, click on "Modify" in the relevant VS
  version; 3) in the newly opened window, select "Individual components" and ensure that v14.24 or above of the MSVC
  "C++ x64/x86 build tools" is selected. The CMake and Boost dependencies are best installed using the latest binary
  executables from the websites of the respective projects.


Source code
'''''''''''

You can check out the latest state of the source code with the command::

    git clone https://github.com/giotto-ai/giotto-tda.git


To install:
'''''''''''

.. code-block:: bash

   cd giotto-tda
   pip install -e ".[tests, doc]"

This way, you can pull the library's latest changes and make them immediately available on your machine.
Note: we recommend upgrading ``pip`` and ``setuptools`` to recent versions before installing in this way.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory::

    pytest gtda


Changelog
---------

See the `RELEASE.rst <https://github.com/giotto-ai/giotto-tda/blob/master/RELEASE.rst>`__ file
for a history of notable changes to giotto-tda.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/giotto-ai/giotto-tda
- Download releases: https://pypi.org/project/giotto-tda/
- Issue tracker: https://github.com/giotto-ai/giotto-tda/issues

Community
---------

giotto-ai Slack workspace: https://slack.giotto.ai/

Contacts
--------

maintainers@giotto.ai
