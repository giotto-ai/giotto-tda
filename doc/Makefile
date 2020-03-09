# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
########## Config

ifndef GITDIR
GITDIR    = ../../../projects/gtda-docs
endif

ifndef VERSION
VERSION = dev
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

togit:
	@mkdir -p "$(CURRENTDOCFOLDER)"
	@cp -a build/html/ "$(CURRENTDOCFOLDER)"

github:
	@make update-versions-list
	@export VERSION=$(VERSION); make html
	@make togit

theory-gl:
	@pandoc theory/glossary.tex -f latex -t rst --toc --include-before-body=theory/before_glossary.rst -s --bibliography=theory/bibliography.bib -o theory/glossary.rst

convert-notebooks:
	@jupyter nbconvert notebooks/*.ipynb --execute --to rst

clean-gh:
	@make clean
	@rm -f theory/glossary.rst
	@make gh-remove

update-versions-list:
	@cd $(GITDIR); find . -type d -depth 1 > versions
	@mv $(GITDIR)/versions versions

all-gh:
	@make clean-gh
	@make theory-gl
	@if ($(RUNNOTEBOOKS)); then\
	   cp ../examples/*.ipynb notebooks/;\
	   cp ../examples/*.py notebooks/;\
	   make convert-notebooks;\
	fi
	@echo Making Docs
	@make github

gh-remove:
	@cd $(GITDIR); rm -rfv $(VERSION) && mkdir $(VERSION)

gh-stage:
	@cd "$(CURRENTDOCFOLDER)"; git add --all;

gh-commit:
	@make gh-stage
	@cd "$(CURRENTDOCFOLDER)"; git commit -a -m "Update docs"; git push origin

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
