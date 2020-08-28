# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages
import os


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='drumgan',
    version='0.0.3',
    description='Gan based drum synth.',
    long_description=readme,
    author='Yotsuyubi',
    author_email='Yotsuyubi@example.com',
    url='https://github.com/Yotsuyubi/drumgan',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=read_requirements(),
    test_suite='tests'
)
