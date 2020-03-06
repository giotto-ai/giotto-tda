############
Contributing
############

.. _contrib:

This page contains a summary of what one needs to do to contribute.

..
   toctree::
   :maxdepth: 2
   :hidden:

   guidelines
   readme_docs

**********
Guidelines
**********

Pull Request Checklist
======================

Before sending your pull requests, make sure you followed this list.
  - Read the `contributing guidelines <https://github.com/giotto-ai/giotto-tda/blob/master/GOVERNANCE.rst>`_.
  - Read the `code of conduct <https://github.com/giotto-ai/giotto-tda/blob/master/CODE_OF_CONDUCT.rst>`_.
  - Ensure you have signed the `contributor license agreement (CLA) <https://cla-assistant.io/giotto-ai/giotto-tda>`_.
  - Check if the changes are consistent with the guidelines.
  - Changes are consistent with the Coding Style.
  - Run Unit Tests.

How to become a contributor and submit your own code
====================================================

Contributor License Agreements
------------------------------

In order to become a contributor of giotto-tda, the first step is to sign the
`contributor license agreement (CLA) <https://cla-assistant.io/giotto-ai/giotto-tda>`_.
**NOTE**: Only original source code from you and other people that have signed
the CLA can be accepted into the main repository.

Contributing code
-----------------

If you have improvements to giotto-tda, do not hesitate to send us pull requests!
Please follow the Github how to (https://help.github.com/articles/using-pull-requests/).
The giotto-tda team will review your pull requests. Once the pull requests are approved and pass continuous integration checks, the
giotto-tda team will work on getting your pull request submitted to our GitHub
repository. Eventually, your pull request will be merged automatically on GitHub.
If you want to contribute, start working through the giotto-tda codebase,
navigate to the `GitHub issue tab <https://github.com/giotto-ai/giotto-tda/issues>`_
and start looking through interesting issues. These are issues that we believe
are particularly well suited for outside contributions, often because we
probably won't get to them right now. If you decide to start on an issue, leave
a comment so that other people know that you're working on it. If you want to
help out, but not alone, use the issue comment thread to coordinate.

Contribution guidelines and standards
=====================================

Before sending your pull request for review, make sure your changes are
consistent with the guidelines and follow the coding style below.

General guidelines and philosophy for contribution
--------------------------------------------------

* Include unit tests when you contribute new features, as they help to
  a) prove that your code works correctly, and
  b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
  usually indicates insufficient test coverage.
* Keep API compatibility in mind when you change code in core giotto-tda.
* Clearly define your exceptions using the utils functions and test the exceptions.
* When you contribute a new feature to giotto-tda, the maintenance burden is   
  (by default) transferred to the giotto-tda team. This means that the benefit   
  of the contribution must be compared against the cost of maintaining the   
  feature.

C++ coding style
----------------

Changes to giotto-tda's C/C++ code should conform to `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_.
Use `clang-tidy` to check your C/C++ changes. To install `clang-tidy` on
ubuntu:16.04, do:


.. code-block:: bash

    apt-get install -y clang-tidy

You can check a C/C++ file by doing:

.. code-block:: bash

    clang-format <my_cc_file> --style=google > /tmp/my_cc_file.ccdiff <my_cc_file> /tmp/my_cc_file.cc

Python coding style
-------------------

Changes to giotto-tda's Python code should conform to PEP8 directives.
Use `flake8` to check your Python changes. To install `flake8` just do

.. code-block:: python

    pip install flake8

You can use `flake8` on your python code via the following instructions:

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

There are two ways to run unit tests for giotto-tda.

1. Using tools and libraries installed directly on your system. The election tool is `pytest`. To install `pytest` just do

.. code-block:: python

    pip install pytest

You can use `pytest` on your python code via the following instructions:

.. code-block:: python

    pytest name_of_your_script.py

2. Using Azure (azure-pipelines.yml) and giotto-tda's CI scripts.


*************
Documentation
*************

Description of the infrastructure
=================================

The documentation is hosted on github-pages (on `https://giotto-ai.github.io/gtda-docs/ <https://giotto-ai.github.io/gtda-docs/>`_).
It has 3 main components:

- API: auto-generated python documentation (sphinx),
- contribution guidelines, README, theory page etc,
- collection of notebook examples (from ../examples, converted to `.py` scripts and executed).

All of the 3 components are `html` pages, generated as described in :ref:`How to build <build_docs>`.
The generated pages have to be committed to a local clone of the
`github repository containing the documentation <https://github.com/giotto-ai/gtda-docs>`_,
which also contains documentation for previous versions of the project.

Generating documentation has been automated and is now part of the Azure CI. Each merge to master generates a new documentation.

How to build
============

.. _build_docs:

Basic instructions
------------------

Create an environment using the ???? file and activate it. We assume that the `giotto-tda` and `gtda-docs` repositories
are cloned in `/path/to/giotto-tda/` and `path/to/gtda-docs/`.

.. code-block:: bash

   cd path/to/giotto-tda/
   cd doc/

Set the environment variable `GITDIR` to `path/to/gtda-docs/`, either directly in the console or editing the `Makefile`:

.. code-block:: bash

   GITDIR = path/to/gtda-docs/

Then, generating the documentation, copying it to the `gtda-docs` repository and committing is done with:

.. code-block:: bash

   make all_gh
   make gh-commit

Note that the success of the `gh-commit` target depends on whether you have writing rights to the `gtda-docs` repository.

Step-by-step instructions
-------------------------

We describe in more details the targets in the Makefile, following the flow described above.

Clean
~~~~~

The `all-gh` target starts with `clean-gh`` which uses the underlying sphinx `clean` target to remove files from the `build` directory.
Similarly, we remove `theory/glossary.rst`, which might have been geenerated by a previous call to `make all-gh`.

Theory
~~~~~~

**This step requires `pandoc` and `pandoc-citeproc` as additional dependencies.**

The glossary `theory/glossary.tex` contains the mathematical definitions of the functionalities
and the terms used throughout the documentation.

The document, along with the bibliography, is converted to an `.rst`, using `pandoc`.
It is included in the main `toctree` of the documentation.
The only purpose and use of `theory/before_glossary.rst` is to have the reference/hyperlink.

Notebooks
~~~~~~~~~

The documentation contains notebooks from `../examples`. They are included as `.rst` files in the `notebooks` directory,
where they are grouped under two categories: `basic` (for quickstarts) and `advanced` (for synthetic examples with some basic analysis).
which record the output.

The workflow with copying the notebooks to `notebooks/`, along with helper functions. Executing them and converting to `.rst with `jupyter nbconvert`
follows in `convert-notebooks`.

It is important to stress that since this step takes a long time, and, for small changes tested locally (without deploying),
we offer the possibility of building the documentation without this step.
To include notebooks in the documentation, set `RUNNOTEBOOKS=TRUE`. This is the default option in the Azure CI. Also, note that building the documentation will probably fail
if the `.rst` files for notebooks are not present.

Html
~~~~

This is the step where `Sphinx` is actually used.
We use the standard `make html` command, which also takes the configuration from `conf.py`.
In particular, we use the `sphinx-rtd-theme <https://github.com/readthedocs/sphinx_rtd_theme>`_.
For details, please see the documentation of `Sphinx` and the extensions listed in `conf.py`.

Move to git and commit
~~~~~~~~~~~~~~~~~~~~~~

Sphinx generates the documentation in the `build` directory. We copy the contents of `build/html/` to `path/to/gtda_docs/$(VERSION)`,
where $(VERSION) is an environment variable (set in the `Makefile`) which dictates the name/tag of the version that we are building.
When opening the page, the user is redirected to the documentation of the latest stable version,
but the documentation for previous versions can still be accessed and is kept for backwards compatibility.

All the changes in the `gtda-docs` repo are staged and committed. A push is tried and requires a password.
It can only succeed if the user has write access to that repository.