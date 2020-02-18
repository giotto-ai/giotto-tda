Description of the infrastructure
=================================

The documentation is hosted on github-pages (https://wreise.github.io/gtda-docs/), from the `../docs/` directory. It has 3 main components:
    - API: auto-generated python documentation (sphinx-numpydoc),
    - contribution guidelines, README, theory page etc,
    - sphinx-gallery (examples from ../examples, converted to `.py` scripts and executed)


Production integration
======================

In this section, we describe the workflow to put the docs online (in production).
 1. The `theory/glossary.tex` is parsed (using pandoc) to a file `theory/glossary.rst`.
 2. Notebooks in `../examples/` are converted to `.py`, which are then executed by the sphinx-gallery extension.
 3. Documentation (html) is generated with `sphinx`, and stored in `build/html/`. This includes parsing files generated in 1. and 2.
 4. The documentation is copied from `build/html/` to `../docs/`, which is a private repository (https://github.com/wreise/gtda-docs) and a submodule of the giotto-tda package.
 5. The contents of the repository are staged, committed and pushed to master.

Steps 1-5 lead to recreating the docs, starting from what is in the cloned repository on the branch `ghpages` (https://github.com/wreise/giotto-learn/tree/ghpages).

How to deploy
==============

The steps mentioned in :ref:`Production Integration` are also reflected in the Makefile. By default, sphinx produces html documentation with the command `make html`, executed from the root of the `doc/` folder.
We have added the following recipes:
- `make theory`: corresponds to step 1 above.
- `make sphinx-gal`: step 2,
- `make github`: steps 3 and 4,
- `make gh-commit`: step 5

The steps `theory, sphinx-gal, github` are encompassed in the `make all-gh` command. To avoid pre-mature publications, we do not provide a single command to also push the documentation to production.


Working with submodules
=======================

While development of modules can be done locally, deploying them requires interaction with the submodule under `../docs/`.

To incorporate the local changes remotely (push the new documentation to gh-pages), use the `gh-commit` target from the Makefile.

To reflect those changes in the main repository (giotto-tda), you need to execute `git submodule add docs/`.

For further reference on the topic, please checkout `https://git-scm.com/book/en/v2/Git-Tools-Submodules`.

