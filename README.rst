.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

|Version|_ |Azure-build|_ |Azure-cov|_ |Azure-test|_ |Twitter-follow|_ |Slack-join|_

.. |Version| image:: https://img.shields.io/pypi/v/giotto-tda
.. _Version:

.. |Azure-build| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-tda?branchName=master
.. _Azure-build: https://dev.azure.com/maintainers/Giotto/_build?definitionId=6&_a=summary&repositoryFilter=6&branchFilter=141&requestedForFilter=ae4334d8-48e3-4663-af95-cb6c654474ea

.. |Azure-cov| image:: https://img.shields.io/azure-devops/coverage/maintainers/Giotto/6/master
.. _Azure-cov:

.. |Azure-test| image:: https://img.shields.io/azure-devops/tests/maintainers/Giotto/6/master
.. _Azure-test:

.. |Twitter-follow| image:: https://img.shields.io/twitter/follow/giotto_ai?label=Follow%20%40giotto_ai&style=social
.. _Twitter-follow: https://twitter.com/intent/follow?screen_name=giotto_ai

.. |Slack-join| image:: https://img.shields.io/badge/Slack-Join-yellow
.. _Slack-join: https://slack.giotto.ai/

giotto-tda
==========

giotto-tda is a high performance topological machine learning toolbox in Python built on top of
scikit-learn and is distributed under the GNU AGPLv3 license. It is part of the `Giotto <https://github.com/giotto-ai>`_ family of open-source projects.

Project genesis
---------------

giotto-tda is the result of a collaborative effort between `L2F SA
<https://www.l2f.ch/>`_, the `Laboratory for Topology and Neuroscience
<https://www.epfl.ch/labs/hessbellwald-lab/>`_ at EPFL, and the `Institute of Reconfigurable & Embedded Digital Systems (REDS)
<https://heig-vd.ch/en/research/reds>`_ of HEIG-VD.

Documentation
-------------

- API reference (stable release): https://docs-tda.giotto.ai
- Theory glossary: https://giotto.ai/theory

Getting started
---------------

To get started with giotto-tda, first follow the installations steps below. `This blog post <https://towardsdatascience.com/getting-started-with-giotto-learn-a-python-library-for-topological-machine-learning-451d88d2c4bc>`_, and references therein, offer a friendly introduction to the topic of topological machine learning and to the philosophy behind giotto-tda.

Tutorials and use cases
~~~~~~~~~~~~~~~~~~~~~~~

Simple tutorials can be found in the `examples <https://github.com/giotto-ai/giotto-tda/tree/master/examples>`_ folder. For a wide selection of use cases and application domains, you can visit `this page <https://giotto.ai/learn/course-content>`_.

Installation
------------

Dependencies
~~~~~~~~~~~~

The latest stable version of giotto-tda requires:

- Python (>= 3.5)
- NumPy (>= 1.17.0)
- SciPy (>= 0.17.0)
- joblib (>= 0.11)
- scikit-learn (>= 0.22.0)
- python-igraph (>= 0.7.1.post6)
- matplotlib (>= 3.0.3)
- plotly (>= 4.4.1)
- ipywidgets (>= 7.5.1)

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

The developer installation requires three important C++ dependencies:

-  A C++14 compatible compiler
-  CMake >= 3.9
-  Boost >= 1.56

Please refer to your system's instructions and to the `CMake <https://cmake.org/>`_ and
`Boost <https://www.boost.org/doc/libs/1_72_0/more/getting_started/index.html>`_ websites for definitive guidance on how to install these dependencies. The instructions below are unofficial, please follow them at your own risk.

Linux
'''''
Most Linux systems should come with a suitable compiler pre-installed. For the other two dependencies, you may consider using your distribution's package manager, e.g. by running

.. code-block:: bash

    sudo apt-get install cmake boost

if ``apt-get`` is available in your system.

macOS
'''''
On macOS, you may consider using ``brew`` (https://brew.sh/) to install the dependencies as follows:

.. code-block:: bash

    brew install gcc cmake boost

Windows
'''''''
On Windows, you will likely need to have `Visual Studio <https://visualstudio.microsoft.com/>`_ installed. At present,
it appears to be important to have a recent version of the VS C++ compiler. One way to check whether this is the case
is as follows: 1) open the VS Installer GUI; 2) under the "Installed" tab, click on "Modify" in the relevant VS
version; 3) in the newly opened window, select "Individual components" and ensure that v14.24 or above of the MSVC
"C++ x64/x86 build tools" is selected. The CMake and Boost dependencies are best installed using the latest binary
executables from the websites of the respective projects.


Source code
'''''''''''

You can obtain the latest state of the source code with the command::

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
