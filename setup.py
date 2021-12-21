# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="TaxoSS",
    version="0.1.4",
    description="Semantic similarity computation with different state-of-the-art metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lorenzo Malandri",
    author_email="lorenzo.malandri@unimib.it",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(include=["TaxoSS"]),
    package_data={'TaxoSS': ['data/*.csv'],},
    include_package_data=False,
    install_requires=["nltk", "numpy", "pandas"]
)