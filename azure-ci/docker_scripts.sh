#!/bin/bash

set -e -x

echo 'files and folders: '
ls
# update-alternatives --set python /opt/python/cp37-cp37m/
/opt/python/*${python_ver}*/bin/pip install --upgrade pip setuptools
#for PYBIN in /opt/python/*/bin; do "${PYBIN}/pip" install cmake; done
/opt/python/*${python_ver}*/bin/pip install cmake
ls /opt/python/*${python_ver}*/bin/
CMAKE_BIN=/opt/python/*${python_ver}*/bin/cmake
ln -sf ${CMAKE_BIN} /usr/bin/cmake

# install boost

#yum list available
yum install -y wget tar
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
echo 'finish downloading boost.'
tar -zxvf /boost_1_69_0.tar.gz
/boost_1_69_0/bootstrap.sh --prefix=/usr/local
find
/b2 install --with=all
ls /usr/local
ls /usr/local/lib
-sBOOST_ROOT=/usr/local/lib

#yum install -y boost148.x86_64
#ls /usr/lib64/
#echo 'other command'
-sBoost_INCLUDE_DIR=/usr/local/lib

/opt/python/cp37-cp37m/bin/pip install -e "/io/.[tests, doc]"
/opt/python/cp37-cp37m/bin/pip uninstall -y giotto-learn
/opt/python/cp37-cp37m/bin/pip install wheel twine

pytest --cov /io/giotto/ --cov-report xml
flake8 --exit-zero /io/
displayName: 'Test with pytest and flake8'

python /io/setup.py sdist bdist_wheel
