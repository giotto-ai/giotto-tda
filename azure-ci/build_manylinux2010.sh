#!/bin/bash
echo 'current directory: '
pwd
ls
docker run -d -i -t --name manylinux10 -v `pwd`:/io quay.io/pypa/manylinux2010_x86_64 /bin/bash
echo 'running dockers:'
docker ps
docker exec manylinux10 ls
docker exec manylinux10 pwd
docker exec manylinux10 sh -c "ls /io"
docker exec manylinux10 sh -c "python --version"
docker exec manylinux10 python --version
docker exec manylinux10 sh -c "python3 --version"
docker exec manylinux10 sh -c "sh -c /io/azure-ci/docker_scripts.sh"
docker exec manylinux10 sh -c "sh /io/azure-ci/docker_scripts.sh"
docker exec manylinux10 sh -c "/io/azure-ci/docker_scripts.sh"
docker exec manylinux10 sudo sh -c "/io/azure-ci/docker_scripts.sh"
