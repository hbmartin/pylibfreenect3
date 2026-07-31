SPHINXOPTS ?= -W --keep-going
BUILDDIR ?= _build

.PHONY: clean html linkcheck

html:
	uv run sphinx-build $(SPHINXOPTS) -b html . $(BUILDDIR)/html

linkcheck:
	uv run sphinx-build $(SPHINXOPTS) -b linkcheck . $(BUILDDIR)/linkcheck

clean:
	uv run python -c 'import shutil; shutil.rmtree("$(BUILDDIR)", ignore_errors=True)'
