#!/bin/bash

set -e -x

echo 'folders: '
ls
/opt/python/cp37-cp37m/bin/pip install --upgrade pip setuptools
/opt/python/cp37-cp37m/bin/pip install cmake --upgrade
/opt/python/cp37-cp37m/bin/pip install -e "/io/.[tests, doc]"
/opt/python/cp37-cp37m/bin/pip uninstall -y giotto-learn
/opt/python/cp37-cp37m/bin/pip install wheel twine

pytest --cov /io/giotto/ --cov-report xml
flake8 --exit-zero /io/
displayName: 'Test with pytest and flake8'

python /io/setup.py sdist bdist_wheel
