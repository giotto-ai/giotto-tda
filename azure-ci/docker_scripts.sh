#!/bin/bash

set -e -x

# upgrading pip and setuptools
PYTHON_PATH=$(eval find "/opt/python/*${python_ver}*" -print)
export PATH=${PYTHON_PATH}/bin:${PATH}
pip install --upgrade pip setuptools

# installing cmake
pip install cmake

# installing boost
yum install -y wget tar
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
tar -zxvf /boost_1_69_0.tar.gz
mkdir boost
cd /boost_1_69_0
./bootstrap.sh --prefix=/boost
./b2 install
cd ..

# helping cmake find boost
export BOOST_ROOT=/boost_1_69_0
export Boost_INCLUDE_DIR=/boost_1_69_0/include

# installing pybind11
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
make install

# installing and uninstalling giotto-learn
cd /io
pip install -e ".[tests]"
pip uninstall -y giotto-learn

# testing, linting
pytest --cov . --cov-report xml
flake8 --exit-zero /io/

# building wheels
pip install wheel twine
python setup.py sdist bdist_wheel
