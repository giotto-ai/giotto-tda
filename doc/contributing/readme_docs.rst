#############
Documentation
#############

*********************************
Description of the infrastructure
*********************************

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

************
How to build
************

.. _build_docs:

Basic instructions
==================

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
=========================

We describe in more details the targets in the Makefile, following the flow described above.

Clean
-----

The `all-gh` target starts with `clean-gh`` which uses the underlying sphinx `clean` target to remove files from the `build` directory.
Similarly, we remove `theory/glossary.rst`, which might have been geenerated by a previous call to `make all-gh`.

Theory
------

**This step requires `pandoc` and `pandoc-citeproc` as additional dependencies.**

The glossary `theory/glossary.tex` contains the mathematical definitions of the functionalities
and the terms used throughout the documentation.

The document, along with the bibliography, is converted to an `.rst`, using `pandoc`.
It is included in the main `toctree` of the documentation.
The only purpose and use of `theory/before_glossary.rst` is to have the reference/hyperlink.

Notebooks
---------

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
----

This is the step where `Sphinx` is actually used.
We use the standard `make html` command, which also takes the configuration from `conf.py`.
In particular, we use the `sphinx-rtd-theme <https://github.com/readthedocs/sphinx_rtd_theme>`_.
For details, please see the documentation of `Sphinx` and the extensions listed in `conf.py`.

Move to git and commit
----------------------

Sphinx generates the documentation in the `build` directory. We copy the contents of `build/html/` to `path/to/gtda_docs/$(VERSION)`,
where $(VERSION) is an environment variable (set in the `Makefile`) which dictates the name/tag of the version that we are building.
When opening the page, the user is redirected to the documentation of the latest stable version,
but the documentation for previous versions can still be accessed and is kept for backwards compatibility.

All the changes in the `gtda-docs` repo are staged and committed. A push is tried and requires a password.
It can only succeed if the user has write access to that repository.