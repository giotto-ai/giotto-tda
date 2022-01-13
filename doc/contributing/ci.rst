################
CI Documentation
################

.. _ci:

This page contains a documentation about the current github actions CI.

..
   toctree::
   :maxdepth: 2
   :hidden:

   ci
   readme_docs

**
CI
**

CI for Pull Request
===================

Pull requests
-------------
 TODO

CI for generating the wheels
============================

The wheel generating is a workflow that needs to be manually triggered. It can be done in the following `Link <https://github.com/giotto-ai/giotto-tda/actions/workflows/wheels.yml>`_. The configuration file of the wheels can be found here ``.github/workflows/wheels.yml``.

The main parts of the workflow are the following:

* Uses `cibuildwheel <https://github.com/pypa/cibuildwheel>`_ to generate the wheels.
* Uses `install-boost <https://github.com/MarkusJx/install-boost>`_ to download boost.
* Uses `cache <https://github.com/actions/cache>`_ cache the boost version to prevent downloading it each time the job is run.

cibuildwheel
------------

For ``cibuildwheel``, some *advanced* features needed to be done, particularly about boost on Linux. The reason why Linux, required special attention is because the ``cibuildwheel`` worker run specifically on `docker images <https://cibuildwheel.readthedocs.io/en/stable/faq/#linux-builds-on-docker>`_. Meaning that, you must provide access to the downloaded boost. As the `documentation <https://cibuildwheel.readthedocs.io/en/stable/faq/#linux-builds-on-docker>`_ states, that a shared folder exist between the host and the docker image, located in ``/host`` for the docker image. In this folder, the entire ``/`` root folder of the host is accessible.

install-boost
-------------

The use of the ``install-boost`` actions, is the same as described in the `README <https://github.com/MarkusJx/install-boost>`_. But two issues were encountered:

1. The action failed to create the custom folder for downloading the archive. To resolve this, a previous step creates manually the folders.
2. On windows, when caching the downloaded files, the next time the job was run and the cache used. Some header files where missing. No reason was found about this behavior, and to "fix" this, on windows, we always download boost.
