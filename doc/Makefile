# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build
GITDIR    = ../docs

togit:
	@cp -a build/html/ "$(GITDIR)"

# prevent the git directory from being cleaned

github:
	@make html
	@make togit

theory:
	@pandoc theory/glossary.tex -f latex -t rst -s --bibliography=theory/bibliography.bib -o theory/glossary.rst

all-gh:
	@make clean
	@make theory
	@make github

gh-commit:
	@cd "$(GITDIR)"; git add *.html generated/*.html;\
	git add objects.inv; git add searchindex.js; \ git add _sources/*.txt; git add _sources/theory/*.txt; git add theory/*.html; \
	git commit -m "Update docs"; git push origin

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
