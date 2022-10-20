#!/usr/bin/env python3

from setuptools import setup, find_packages
from pyvib import __version__

def readme():
    with open('README.org') as f:
        return f.read()


setup(name='pyvib',
      version=__version__,
      # use find_packages to include subfolders with a __init__.py
      # https://stackoverflow.com/a/43254082
      packages=find_packages(),
      description='Nonlinear modeling for python',
      long_description=readme(),
      classifiers=[],
      keywords=['nonlinear','identification','bifurcation','simulation'],
      url='https://github.com/pawsen/pyvib',
      author='Paw Møller',
      author_email='pawsen@gmail.com',
      license='BSD',
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib'],
      extras_require={
          'tests': [
              #'nose',
              #'pycodestyle >= 2.1.0'
          ],
          'docs': [
              'sphinx >= 1.4',
              'sphinx_rtd_theme']}
)
