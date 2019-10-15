#!/bin/bash
echo 'current directory: '
docker run -d -i -t --name manylinux10 -v `pwd`:/io quay.io/pypa/manylinux2010_x86_64 /bin/bash
docker exec manylinux10 sh -c "sh /io/azure-ci/docker_scripts.sh"
