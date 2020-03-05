**********************
Developer installation
**********************

.. _dev_installation:

Installing both the PyPI release and source of giotto-tda in the same environment is not recommended since it is
known to cause conflicts with the C++ bindings.

The developer installation requires three important C++ dependencies:

-  A C++14 compatible compiler
-  CMake >= 3.9
-  Boost >= 1.56

Please refer to your system's instructions and to the `CMake <https://cmake.org/>`_ and
`Boost <https://www.boost.org/doc/libs/1_72_0/more/getting_started/index.html>`_ websites for definitive guidance on how to install these dependencies. The instructions below are unofficial, please follow them at your own risk.

Linux
=====

Most Linux systems should come with a suitable compiler pre-installed. For the other two dependencies, you may consider using your distribution's package manager, e.g. by running

.. code-block:: bash

    sudo apt-get install cmake libboost-dev

if ``apt-get`` is available in your system.

macOS
=====

On macOS, you may consider using ``brew`` (https://brew.sh/) to install the dependencies as follows:

.. code-block:: bash

    brew install gcc cmake boost

Windows
=======

On Windows, you will likely need to have `Visual Studio <https://visualstudio.microsoft.com/>`_ installed. At present,
it appears to be important to have a recent version of the VS C++ compiler. One way to check whether this is the case
is as follows: 1) open the VS Installer GUI; 2) under the "Installed" tab, click on "Modify" in the relevant VS
version; 3) in the newly opened window, select "Individual components" and ensure that v14.24 or above of the MSVC
"C++ x64/x86 build tools" is selected. The CMake and Boost dependencies are best installed using the latest binary
executables from the websites of the respective projects.


Source code
===========

You can obtain the latest state of the source code with the command::

    git clone https://github.com/giotto-ai/giotto-tda.git


To install:
===========

.. code-block:: bash

   cd giotto-tda
   python -m pip install -e ".[dev]"

This way, you can pull the library's latest changes and make them immediately available on your machine.
Note: we recommend upgrading ``pip`` and ``setuptools`` to recent versions before installing in this way.

Testing
=======

After installation, you can launch the test suite from outside the
source directory::

    pytest gtda

