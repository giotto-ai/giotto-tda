#!/bin/bash
set -e
docker run -t --rm -e python_ver=$PYTHON_VER \
	-v `pwd`:/io \
	-v "${CCACHE_DIR}":/root/.ccache/  \
	quay.io/pypa/manylinux2010_x86_64:2020-12-03-912b0de \
	/bin/bash -c "bash /io/.azure-ci/docker_scripts.sh"
