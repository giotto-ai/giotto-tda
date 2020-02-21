# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
GITDIR    = ../../../projects/gtda-docs
EXAMPLESDIR  = ../examples
VERSION = stable

CURRENTDOCFOLDER = $(GITDIR)/$(VERSION)

togit:
	@mkdir -p "$(CURRENTDOCFOLDER)"
	@cp -a build/html/ "$(CURRENTDOCFOLDER)"

github:
	@make html
	@make togit
	@python create_index.py

theory:
	@pandoc theory/glossary.tex -f latex -t rst -s --bibliography=theory/bibliography.bib -o theory/glossary.rst

all-gh:
	@make clean
	@make theory
	@make sphinx-gal
	@make github

gh-commit:
	@cd "$(GITDIR)"; git commit -a -m "Update docs"

sphinx-gal:
	@for f in $(EXAMPLESDIR)/*.ipynb; do python3 ipynb_to_py.py $$f ; done

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
