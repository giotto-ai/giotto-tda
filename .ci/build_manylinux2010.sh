#!/bin/bash

docker run -d -i -t --name manylinux10 -v `pwd`:/io quay.io/pypa/manylinux2010_x86_64 /bin/bash
echo 'running dockers:'
docker ps
docker exec manylinux10 ls
docker exec manylinux10 pwd
docker exec manylinux10 sh -c "ls /io"
docker exec manylinux10 sh -c "python --version"
docker exec manylinux10 python --version
docker exec manylinux10 sh -c "python3 --version"
docker exec manylinux10 sh -c "sudo sh -c /io/.ci/build_manylinux2010.sh"
docker exec manylinux10 sh -c "sudo sh /io/.ci/build_manylinux2010.sh"
docker exec manylinux10 sh -c "sudo bash /io/.ci/build_manylinux2010.sh"


set -e -x

echo 'folders: '
ls
python -m pip install --upgrade pip setuptools

pip install -e /io/.['tests', 'doc']
pip uninstall -y giotto-learn
pip install wheel twine

pytest --cov /io/giotto/ --cov-report xml
flake8 --exit-zero /io/
displayName: 'Test with pytest and flake8'

python /io/setup.py sdist bdist_wheel
