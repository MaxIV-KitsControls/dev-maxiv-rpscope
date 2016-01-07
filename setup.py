#!/usr/bin/env python

# Imports
import os
from setuptools import setup


# Read function
def safe_read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except IOError:
        return ""

# Setup
setup(name="tangods-rpscope",
      version="0.1.0",
      py_modules=['rpscope'],
      entry_points={
          'console_scripts': ['RpScope = rpscope:run']},

      license="GPLv3",
      description="Red Pitaya scope device server",
      long_description=safe_read("README.md"),

      author="Vincent Michel",
      author_email="vincent.michel@maxlab.lu.se",
      url="http://www.maxlab.lu.se",
      )
