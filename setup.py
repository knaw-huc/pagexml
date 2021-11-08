#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as f:
    readme = f.read()

setup(
    name='pagexml',
    version='0.0.2',

    description='Utility functions for reading PageXML files',
    long_description=readme,
    long_description_content_type='text/markdown',

    author='Marijn Koolen <marijn.koolen@huygens.knaw.nl>, Bram Buitendijk <bram.buitendijk@di.huc.knaw.nl>',
    author_email='bram.buitendijk@di.huc.knaw.nl',

    url='https://github.com/knaw-huc/pagexml',
    license='MIT',

    packages=find_packages(exclude=('tests', 'docs')),

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
    ],
)
