#!/bin/bash
set -e
set -x
echo "Start manylinux2010 docker build"

# Upgrade pip and setuptools. TODO: Monitor status of pip versions
PYTHON_PATH=$(eval find "/opt/python/*${python_ver}*" -print)
export PATH="${PYTHON_PATH}/bin:${PATH}"
pip config set global.progress_bar off
pip install --upgrade pip setuptools

# Install CMake
pip install cmake

# workaround CentOS 6.10 that reached EOL
# https://github.com/pypa/manylinux/issues/836
sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf
sed -i 's/mirrorlist/#mirrorlist/' /etc/yum.repos.d/CentOS-*.repo
sed -i 's/#\(baseurl.*\)mirror.centos.org/\1vault.centos.org/' /etc/yum.repos.d/CentOS-Base.repo
rm -f /etc/yum.repos.d/CentOS-SCLo-scl*

yum makecache

# Setup ccache
yum install -y ccache
source /io/.azure-ci/setup_ccache.sh

ccache -s

# Install boost
yum install -y wget tar
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
tar -zxvf /boost_1_69_0.tar.gz
mkdir boost
cd /boost_1_69_0
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
