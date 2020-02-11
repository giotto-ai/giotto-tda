#!/bin/bash
docker run -i -t -e python_ver=$PYTHON_VER --name manylinux10 \
	-v `pwd`:/io quay.io/pypa/manylinux2010_x86_64 \
	-v "${CCACHE_DIR}":/root/.ccache/  \
	/bin/bash -c "bash /io/.azure-ci/docker_scripts.sh"
