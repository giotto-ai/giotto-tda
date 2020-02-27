# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
########## Config

GITDIR    = ../../../projects/gtda-docs
VERSION = stable

########## Constant config
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
EXAMPLESDIR  = ../examples

CURRENTDOCFOLDER = $(GITDIR)/$(VERSION)

togit:
	@mkdir -p "$(CURRENTDOCFOLDER)"
	@cp -a build/html/ "$(CURRENTDOCFOLDER)"

github:
	@make html
	@make togit
	@cd $(GITDIR); python create_index.py

theory-gl:
	@pandoc theory/glossary.tex -f latex -t rst --reference-location=block -s --bibliography=theory/bibliography.bib -o theory/glossary.rst

clean-gh:
	@make clean
	@rm theory/glossary.rst

all-gh:
	@make clean-gh
	@make theory-gl
	@cp ../examples/*.ipynb notebooks/
	@cp ../examples/plotting.py notebooks/
	@make github

gh-commit:
	@cd "$(CURRENTDOCFOLDER)"; git add *; git commit -a -m "Update docs"; git push origin

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
