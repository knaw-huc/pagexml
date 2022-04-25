This project uses [Poetry](https://python-poetry.org/) for dependency management and more.

When updating the version, please use the custom `version` function from [`poetry_scripts.py`](poetry_scripts.py):

```
poetry run version [ patch | minor | major | prepatch | preminor | premajor | prerelease | <valid semver string> ] 
```

or from within a `poetry shell` just

```
version [ patch | minor | major | prepatch | preminor | premajor | prerelease | <valid semver string> ] 
```

instead of the default 

```
poetry version
```

as `poetry run version` also updates the `__version__` variable in [`annorepo/__init__.py`](annorepo/__init__.py)