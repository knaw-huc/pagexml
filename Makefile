.PHONY: docs help clean tests

all: help

docs:
	cd docs && rm -rf generated && make clean && make html

clean:
	cd docs && rm -rf generated && make clean

tests:
	poetry run python -m unittest tests/*_test.py

poetry.lock: pyproject.toml
	poetry update --lock

requirements.txt: poetry.lock
	poetry export -o requirements.txt --without-hashes

help:
	@echo "make-tools for pagexml-tools"
	@echo
	@echo "Please use \`make <target>', where <target> is one of:"
	@echo "  docs           	to build or update the documentation pages in docs/_build"
	@echo "  clean          	to remove all generated files and directories"
	@echo "  tests          	to run the unit tests in tests/"
	@echo "  requirements.txt	to update the requirements.txt based on poetry.lock (required by readthedocs)"
