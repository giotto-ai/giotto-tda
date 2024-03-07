############
Contributing
############

.. _contrib:

This page contains a summary of what one needs to do to contribute.

.. toctree::
   :maxdepth: 2
   :hidden:

   guidelines
   ci

**********
Guidelines
**********

Essentials for contributing
===========================

Contributor License Agreement
-----------------------------

In order to become a contributor of ``giotto-tda``, the first step is to sign the
`contributor license agreement (CLA) <https://cla-assistant.io/giotto-ai/giotto-tda>`_.
**NOTE**: Only original source code from you and other people that have signed
the CLA can be accepted into the main repository.

Pull requests
-------------

If you have improvements to ``giotto-tda``, do not hesitate to send us pull requests!
Please follow the `Github how to <https://help.github.com/articles/using-pull-requests>`_ and
make sure you followed this checklist *before* submitting yor pull request:

- Make sure you have signed the `contributor license agreement (CLA) <https://cla-assistant.io/giotto-ai/giotto-tda>`_.
- Read the :ref:`Contribution guidelines and standards <contribution_guidelines_standards>`.
- Read the `code of conduct <https://github.com/giotto-ai/giotto-tda/blob/master/CODE_OF_CONDUCT.rst>`_.
- Check that the changes are consistent with the guidelines and coding styles.
- Run unit tests.

The ``giotto-tda`` team will review your pull requests. Once the pull requests are approved
and pass continuous integration checks, the ``giotto-tda`` team will work on getting your pull
request submitted to our GitHub repository. Eventually, your pull request will be merged
automatically on GitHub.

Issues
------

If you would like to know how you can contribute to the ``giotto-tda`` codebase, we recommend
that you navigate to the `GitHub issue tab <https://github.com/giotto-ai/giotto-tda/issues>`_
and start looking through interesting issues. If you decide to start working on an issue, leave
a comment so that other people know that you're working on it. If you want to help out, but not
alone, use the issue comment thread to coordinate.

Contribution guidelines and standards
=====================================

.. _contribution_guidelines_standards:

Before sending your pull request for review, make sure your changes are
consistent with the guidelines and follow the coding style below.

General guidelines and philosophy for contribution
--------------------------------------------------

* Include unit tests when you contribute new features, as they help to
  a) prove that your code works correctly, and
  b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
  usually indicates insufficient test coverage.
* Keep API compatibility in mind when you change code in core ``giotto-tda``.
* Clearly define your exceptions using the utils functions and test the exceptions.
* When you contribute a new feature to ``giotto-tda``, the maintenance burden is   
  (by default) transferred to the ``giotto-tda`` team. This means that the benefit   
  of the contribution must be compared against the cost of maintaining the feature.

C++ coding style
----------------

Changes to ``giotto-tda``'s C/C++ code should conform to `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_.
Use ``clang-tidy`` to check your C/C++ changes. As an example, to install ``clang-tidy`` on Ubuntu 16.04, do:


.. code-block:: bash

    apt-get install -y clang-tidy

You can check a C/C++ file by running:

.. code-block:: bash

    clang-format <my_cc_file> --style=google > /tmp/my_cc_file.ccdiff <my_cc_file> /tmp/my_cc_file.cc

Python coding style
-------------------

Whenever possible, changes to ``giotto-tda``'s Python code should conform to
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ directives. Use ``flake8`` to check your Python
changes. To install ``flake8`` just do

.. code-block:: python

    python -m pip install flake8

You can use ``flake8`` on your python code via the following instructions:

.. code-block:: python

    flake8 name_of_your_script.py

Git pre-commit hook
-------------------
We provide a pre-commit git hook to prevent accidental commits to the master branch. To activate, run

.. code-block:: bash

    cd .git/hooks
    ln -s ../../.tools/git-pre-commit pre-commit

Running unit tests
------------------

There are two ways to run unit tests for ``giotto-tda``.

1. Using tools and libraries installed directly on your system. ``giotto-tda`` relies on ``pytest``.
   To install ``pytest`` just run

.. code-block:: python

    python -m pip install pytest

You can use ``pytest`` on your python code via the following instructions:

.. code-block:: python

    pytest name_of_your_script.py

2. Using Azure and ``giotto-tda``'s `CI scripts <https://github.com/giotto-ai/giotto-tda/blob/master/azure-pipelines.yml>`_.