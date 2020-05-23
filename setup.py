#!/usr/bin/env python

from setuptools import setup

setup(name='rrcf',
      version='0.4.1',
      description='Robust random cut forest for anomaly detection',
      author='Matt Bartos, Abhiram Mullapudi, Sara Troutman',
      author_email='mdbartos@umich.edu, abhiramm@umich.edu, stroutm@umich.edu',
      url='http://open-storm.org',
      packages=["rrcf"],
      install_requires=[
          'numpy'
      ]
      )
