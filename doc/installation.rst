############
Installation
############

.. _installation:

************
Dependencies
************

The latest stable version of giotto-tda requires:

- Python (>= 3.6)
- NumPy (>= 1.19.1)
- SciPy (>= 1.5.0)
- joblib (>= 0.16.0)
- scikit-learn (>= 0.23.1)
- pyflagser (>= 0.4.0)
- python-igraph (>= 0.8.2)
- plotly (>= 4.8.2)
- ipywidgets (>= 7.5.1)

To run the examples, jupyter is required.


*****************
User installation
*****************

The simplest way to install giotto-tda is using ``pip``   ::

    python -m pip install -U giotto-tda

If necessary, this will also automatically install all the above dependencies. Note: we recommend
upgrading ``pip`` to a recent version as the above may fail on very old versions.

Pre-release, experimental builds containing recently added features, and/or
bug fixes can be installed by running   ::

    python -m pip install -U giotto-tda-nightly

The main difference between giotto-tda-nightly and the developer installation (see the section
on contributing, below) is that the former is shipped with pre-compiled wheels (similarly to the stable
release) and hence does not require any C++ dependencies. As the main library module is called ``gtda`` in
both the stable and nightly versions, giotto-tda and giotto-tda-nightly should not be installed in
the same environment.

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
is as follows:

1. open the VS Installer GUI;
2. under the "Installed" tab, click on "Modify" in the relevant VS version;
3. in the newly opened window, select "Individual components" and ensure that v14.24 or above of the MSVC "C++ x64/x86 build tools" is selected. The CMake and Boost dependencies are best installed using the latest binary executables from the websites of the respective projects.

Boost
-----

Some users are experiencing some issue when installation `boost` on Windows. To help them resolve this issue, we customized a little bit the detection of `boost` library.
To install boost on windows, we (maintainers of giotto-tda) recommend 3 options:

- Pre-built binaries,
- Directly from source,
- Use an already installed boost version that fulfills `giotto-tda` requirements.

Pre-built binaries
------------------

Boost propose for windows pre-built binaries to ease the installation of boost
in your system. In the
`website <https://sourceforge.net/projects/boost/files/boost-binaries/>`_, you'll have access to all versions of boost. At the time of writing
this documentation, the most recent version of boost is `1.72.0`. If you go
into the folder, you'll find different executables - choose the version
corresponding to your system (32, 64 bits). In our case, we downloaded `boost_1_72_0-msvc-14.2-64.exe`.
Follow the installation instructions, and when prompted to specify the folder to install boost, go for `C:\\local\\`.

Source code
-----------

Boost proposes to `download <https://www.boost.org/users/download/>`_ directly the source code of boost.
You can choose from different sources (compressed in `.7z` or `.zip`).
Download one and uncompress it in `C:\\local\\`, so you should have something like `C:\\local\\boost_x_y_z\\<boost_files>`.

Already installed boost version
-------------------------------

If by some obscure reason, you have boost installed in your system but the installation procedure cannot find it (can happen, no control on cmake ...).
You can help the installation script by adding the path to your installation in the following place `gtda\\cmake\\HelperBoost.cmake`.
In `HelperBoost.cmake` file, line 7, you can add your path between the quotation marks, e.g.::

   list(APPEND BOOST_ROOT "C:\\<path_to_your_boost_installation>").

Troubleshooting
---------------

If you need to understand where the compiler tries to look for ``boost`` headers,
you can install ``giotto-tda`` with::

   python -m pip install -e . -v

Then you can look at the output for lines starting with::

   Boost_INCLUDE_DIR: <path>
   Boost_INCLUDE_DIRS: <path>

Also, if you have installed different versions of ``boost`` in the process of trying to instal ``giotto-tda``,
make sure to clear CMake cache entries::

    rm -rf build/


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
