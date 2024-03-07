#!/usr/bin/env python

"""Package setup file.
"""

from setuptools import setup, find_packages

setup(
    name='nemo',
    version='1.0',
    description='Neural Elevation Models',
    author='Adam Dai',
    author_email='adamdai97@gmail.com',
    url='https://github.com/adamdai/neural_elevation_models',
    packages=find_packages(),
)