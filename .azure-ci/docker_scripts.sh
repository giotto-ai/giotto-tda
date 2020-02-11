#!/bin/bash
set -x
echo "Start manylinux2010 docker build"

# Upgrading pip and setuptools, TODO: Monitor status of pip versions
PYTHON_PATH=$(eval find "/opt/python/*${python_ver}*" -print)
export PATH=${PYTHON_PATH}/bin:${PATH}
pip install --upgrade pip==19.3.1 setuptools

# Install CMake
pip install cmake

# Setup ccache
yum install -y ccache
mkdir /ccache
ln -s /usr/bin/ccache /ccache/gcc
ln -s /usr/bin/ccache /ccache/g++
ln -s /usr/bin/ccache /ccache/cc
ln -s /usr/bin/ccache /ccache/c++
export PATH=/ccache/:${PATH}
# maximum cache size and compression
ccache -M 1024M
export CCACHE_COMPRESS=1

ccache -s

# Install boost
yum install -y wget tar
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
tar -zxvf /boost_1_69_0.tar.gz
mkdir boost
cd /boost_1_69_0
./bootstrap.sh --prefix=/boost
./b2 install -j3
cd ..

ccache -s

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
