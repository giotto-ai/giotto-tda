#!/bin/bash
set -x

# Upgrade pip and setuptools. TODO: Monitor status of pip versions
PYTHON_PATH=$(eval find "/opt/python/*${python_ver}*" -print)
export PATH=${PYTHON_PATH}/bin:${PATH}
pip install --upgrade pip==19.3.1 setuptools

# Install CMake
pip install cmake

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

# Install dev environment
cd /io
pip install -e ".[tests, doc]"

# Test dev install with pytest and flake8
pytest gtda --cov --cov-report xml
flake8 --exit-zero /io/

# Uninstall giotto-tda/giotto-tda-nightly dev
pip uninstall -y giotto-tda
pip uninstall -y giotto-tda-nightly

# Build wheels
pip install wheel
python setup.py sdist bdist_wheel
