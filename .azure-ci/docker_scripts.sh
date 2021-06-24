#!/bin/bash
set -e
set -x
echo "Start manylinux2010 docker build"

# Upgrade pip and setuptools. TODO: Monitor status of pip versions
PYTHON_PATH=$(eval find "/opt/python/*cp${python_ver}*" -print)
export PATH="${PYTHON_PATH}/bin:${PATH}"
pip config set global.progress_bar off
pip install --upgrade pip setuptools

# Install CMake
pip install cmake

# Setup ccache
yum install -y ccache
source /io/.azure-ci/setup_ccache.sh

ccache -s

# Install boost
yum install -y wget tar
wget --no-check-certificate https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
tar -zxvf /boost_1_76_0.tar.gz
mkdir boost
cd /boost_1_76_0
./bootstrap.sh --prefix=/boost
./b2 install -j3 || echo "Parts of boost failed to build. Continuing..."
cd ..

ccache -s

# Help CMake find boost
export BOOST_ROOT=/boost
export Boost_INCLUDE_DIR=/boost/include

# Install dev environment
cd /io
pip install wheel
pip install -e ".[dev]"

# Test dev install with pytest
pytest gtda --no-cov --no-coverage-upload

# Uninstall giotto-tda/giotto-tda-nightly dev
pip uninstall -y giotto-tda
pip uninstall -y giotto-tda-nightly

# Build wheels
python setup.py bdist_wheel

# Repair wheels with auditwheel
pip install auditwheel
auditwheel repair dist/*whl -w dist/
# remove wheels that are not manylinux2010
rm -rf dist/*-linux*.whl
