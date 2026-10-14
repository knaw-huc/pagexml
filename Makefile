.PHONY: docs docs-source help clean test install publish version-update-patch version-update-minor version-update-major

all: help

SHELL = /bin/bash
RED = \033[1;31m
GREEN = \033[1;32m
YELLOW = \033[1;33m
BLUE = \033[1;34m
RESET = \033[0m
TAG=pagexml-tools

docs:
	cd docs && rm -rf generated && make clean && make html

apidoc-doc-source:
	sphinx-apidoc -o docs/source/api --separate --force --module-first pagexml

clean:
	cd docs && rm -rf generated && make clean

test:
	poetry run python -m unittest tests/*_test.py

poetry.lock: pyproject.toml
	poetry update --lock

docs/requirements.txt: poetry.lock
	poetry export -o docs/requirements.txt --without-hashes

install:
	poetry lock && poetry install

publish:
	poetry build && poetry publish

version-update-patch:
	poetry run version patch

version-update-minor:
	poetry run version minor

version-update-major:
	poetry run version major

help:
	@echo -e "make-tools for $(GREEN)$(TAG)$(RESET)"
	@echo
	@echo -e "Please use \`$(YELLOW)make <target>$(RESET)', where $(YELLOW)<target>$(RESET) is one of:"
	@echo -e "  $(BLUE)install$(RESET)               - to install the necessary requirements"
	@echo -e "  $(BLUE)apidoc-source$(RESET)         - to build or update the apidoc source files in docs/source/api"
	@echo -e "  $(BLUE)docs$(RESET)                  - to build or update the documentation pages in docs/_build"
	@echo -e "  $(BLUE)clean$(RESET)                 - to remove all generated files and directories"
	@echo -e "  $(BLUE)test$(RESET)                  - to run the unit tests in tests/"
	@echo -e "  $(BLUE)version-update-patch$(RESET)  - to update the project version to the next patch version"
	@echo -e "  $(BLUE)version-update-minor$(RESET)  - to update the project version to the next minor version"
	@echo -e "  $(BLUE)version-update-major$(RESET)  - to update the project version to the next major version"
	@echo -e "  $(BLUE)publish$(RESET)               - to publish to pypi"
	@echo -e "  $(BLUE)docs/requirements.txt$(RESET) - to update the requirements.txt based on poetry.lock (required by readthedocs)"
