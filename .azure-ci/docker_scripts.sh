#!/bin/bash

set -x

# Upgrading pip and setuptools, TODO: Monitor status of pip versions
PYTHON_PATH=$(eval find "/opt/python/*${python_ver}*" -print)
export PATH=${PYTHON_PATH}/bin:${PATH}
pip install --upgrade pip==19.3.1 setuptools

# Install CMake
pip install cmake

# Install dependencies for python-igraph
yum install -y libxml2 libxml2-devel zlib1g-devel bison flex

# Install boost
yum install -y wget tar
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
tar -zxvf /boost_1_69_0.tar.gz
mkdir boost
cd /boost_1_69_0
./bootstrap.sh --prefix=/boost
./b2 install
cd ..

# Help CMake find boost
export BOOST_ROOT=/boost
export Boost_INCLUDE_DIR=/boost/include

# Install and uninstall giotto-tda dev
cd /io
pip install -e ".[tests, doc]"
pip uninstall -y giotto-tda
pip uninstall -y giotto-tda-nightly

# Testing, linting
pytest --cov . --cov-report xml
flake8 --exit-zero /io/

# Building wheels
pip install wheel
python setup.py sdist bdist_wheel
