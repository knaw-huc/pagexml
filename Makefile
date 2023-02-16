.PHONY: docs help clean tests

all: help

docs:
	cd docs && rm -rf generated && make clean && make html

clean:
	cd docs && rm -rf generated && make clean

tests:
	poetry run python -m unittest tests/*_test.py

help:
	@echo "make-tools for pagexml-tools"
	@echo
	@echo "Please use \`make <target>', where <target> is one of:"
	@echo "  docs           to build or update the documentation pages in docs/_build"
	@echo "  clean          to remove all generated files and directories"
	@echo "  tests          to run the unit tests in tests/"
