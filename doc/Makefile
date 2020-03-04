# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
########## Config

ifndef GITDIR
GITDIR    = ../../../projects/gtda-docs
endif

ifndef VERSION
VERSION = stable
endif

ifndef RUNNOTEBOOKS
RUNNOTEBOOKS=FALSE
endif

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
	@pandoc theory/glossary.tex -f latex -t rst --toc --include-before-body=theory/before_glossary.rst -s --bibliography=theory/bibliography.bib -o theory/glossary.rst

convert-notebooks:
	@jupyter nbconvert notebooks/*.ipynb --execute --to rst

clean-gh:
	@make clean
	@rm -f theory/glossary.rst

all-gh:
	@make clean-gh
	@make theory-gl
	@if ($(RUNNOTEBOOKS)); then\
	   cp ../examples/*.ipynb notebooks/;\
	   cp ../examples/*.py notebooks/;\
	   make convert-notebooks;\
	fi
	@make github

gh-stage:
	@cd "$(CURRENTDOCFOLDER)"; git add --all;

gh-commit:
	@make gh-stage
	@cd "$(CURRENTDOCFOLDER)"; git commit -a -m "Update docs"; git push origin

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
