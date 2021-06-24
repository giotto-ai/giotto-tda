.. image:: https://raw.githubusercontent.com/giotto-ai/giotto-tda/master/doc/images/tda_logo.svg
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

==========
giotto-tda
==========

``giotto-tda`` is a high-performance topological machine learning toolbox in Python built on top of
``scikit-learn`` and is distributed under the GNU AGPLv3 license. It is part of the `Giotto <https://github.com/giotto-ai>`_
family of open-source projects.

Project genesis
===============

``giotto-tda`` is the result of a collaborative effort between `L2F SA <https://www.l2f.ch/>`_,
the `Laboratory for Topology and Neuroscience <https://www.epfl.ch/labs/hessbellwald-lab/>`_ at EPFL,
and the `Institute of Reconfigurable & Embedded Digital Systems (REDS) <https://heig-vd.ch/en/research/reds>`_ of HEIG-VD.

License
=======

.. _L2F team: business@l2f.ch

``giotto-tda`` is distributed under the AGPLv3 `license <https://github.com/giotto-ai/giotto-tda/blob/master/LICENSE>`_.
If you need a different distribution license, please contact the `L2F team`_.

Documentation
=============

Please visit `https://giotto-ai.github.io/gtda-docs <https://giotto-ai.github.io/gtda-docs>`_ and navigate to the version you are interested in.

Installation
============

Dependencies
------------

The latest stable version of ``giotto-tda`` requires:

- Python (>= 3.6)
- NumPy (>= 1.19.1)
- SciPy (>= 1.5.0)
- joblib (>= 0.16.0)
- scikit-learn (>= 0.23.1)
- pyflagser (>= 0.4.3)
- python-igraph (>= 0.8.2)
- plotly (>= 4.8.2)
- ipywidgets (>= 7.5.1)

To run the examples, jupyter is required.

User installation
-----------------

The simplest way to install ``giotto-tda`` is using ``pip``   ::

    python -m pip install -U giotto-tda

If necessary, this will also automatically install all the above dependencies. Note: we recommend
upgrading ``pip`` to a recent version as the above may fail on very old versions.

Pre-release, experimental builds containing recently added features, and/or
bug fixes can be installed by running   ::

    python -m pip install -U giotto-tda-nightly

The main difference between ``giotto-tda-nightly`` and the developer installation (see the section
on contributing, below) is that the former is shipped with pre-compiled wheels (similarly to the stable
release) and hence does not require any C++ dependencies. As the main library module is called ``gtda`` in
both the stable and nightly versions, ``giotto-tda`` and ``giotto-tda-nightly`` should not be installed in
the same environment.

Developer installation
----------------------

Please consult the `dedicated page <https://giotto-ai.github.io/gtda-docs/latest/installation.html#developer-installation>`_
for detailed instructions on how to build ``giotto-tda`` from sources across different platforms.

.. _contributing-section:

Contributing
============

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to ``giotto-tda``, please consult `the relevant page
<https://giotto-ai.github.io/gtda-docs/latest/contributing/index.html>`_.

Testing
-------

After developer installation, you can launch the test suite from outside the
source directory   ::

    pytest gtda

Important links
===============

- Official source code repo: https://github.com/giotto-ai/giotto-tda
- Download releases: https://pypi.org/project/giotto-tda/
- Issue tracker: https://github.com/giotto-ai/giotto-tda/issues


Citing giotto-tda
=================

If you use ``giotto-tda`` in a scientific publication, we would appreciate citations to the following paper:

   `giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration <https://arxiv.org/abs/2004.02551>`_, Tauzin *et al*, arXiv:2004.02551, 2020.

You can use the following BibTeX entry:

.. code:: RST

    @misc{tauzin2020giottotda,
          title={giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration},
          author={Guillaume Tauzin and Umberto Lupo and Lewis Tunstall and Julian Burella Pérez and Matteo Caorsi and Anibal Medina-Mardones and Alberto Dassatti and Kathryn Hess},
          year={2020},
          eprint={2004.02551},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

Community
=========

giotto-ai Slack workspace: https://slack.giotto.ai/

Contacts
========

maintainers@giotto.ai
