# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
########## Config

ifndef GITDIR
GITDIR    = ../../../Projects/gtda-docs
endif

ifndef VERSION
VERSION = latest
endif

ifndef RUNNOTEBOOKS
RUNNOTEBOOKS = FALSE
endif

########## Constant config
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
EXAMPLESDIR  = ../examples

SHOW_VERSIONS = TRUE
PRODUCTION_DOMAIN = giotto-ai.github.io/gtda-docs/

CURRENTDOCFOLDER = $(GITDIR)/$(VERSION)

REQUIRED="pandoc pandoc-citeproc jupyter"

check_installs:
	@for cmd in ${REQUIRED};\
	do
		if ! command -v ${cmd} 2> /dev/null;\
		then
			echo Please install ${cmd}!;\
			exit 1
		fi;\
	done;

copy-to-docs-location:
	@mkdir -p "$(CURRENTDOCFOLDER)"
	@cp -r build/html/* "$(CURRENTDOCFOLDER)"

test-notebooks:
	@make run-notebooks
	@make html-docs

html-docs:
	@export VERSION=$(VERSION); make html

theory-gl:
	@pandoc theory/glossary.tex -f latex -t rst --toc -s --bibliography=theory/bibliography.bib -o theory/glossary.rst

clean-gh:
	@make clean
	@rm -f theory/glossary.rst
	@make gh-remove

update-versions:
	@export VERSION=$(VERSION); python3 update_versions.py $(GITDIR)

update-versions-list:
	@cd $(GITDIR); find . -type d -depth 1 > versions
	@mv $(GITDIR)/versions versions
	@make update-versions

run-notebooks:
	@cp ../examples/*.ipynb notebooks/
	@cp -r ../examples/data notebooks/
	@cp -r ../examples/images notebooks/
	@jupyter nbconvert notebooks/*.ipynb --execute --to rst --ExecutePreprocessor.timeout=1200

all-gh:
	@make clean-gh
	@make theory-gl
	@if ($(RUNNOTEBOOKS)); then\
	   make run-notebooks;\
	fi
	@if ($(UPDATE_VERSIONS)); then\
	    make update-versions-list;\
	fi
	@make html-docs
	@make copy-to-docs-location

gh-remove:
	@cd $(GITDIR); rm -rfv $(VERSION) && mkdir $(VERSION)

gh-stage:
	@cd "$(CURRENTDOCFOLDER)"; git add --all;

gh-commit:
	@make gh-stage
	@cd "$(CURRENTDOCFOLDER)"; git commit -a -m "Update docs";

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
