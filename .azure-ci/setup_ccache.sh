#!/bin/sh
echo "Setting up ccache"
mkdir /tmp/ccache/
ln -s $(which ccache) /tmp/ccache/gcc
ln -s $(which ccache) /tmp/ccache/g++
ln -s $(which ccache) /tmp/ccache/cc
ln -s $(which ccache) /tmp/ccache/c++
export PATH="/tmp/ccache/:${PATH}"
# maximum cache size and compression
ccache -M 1024M
export CCACHE_COMPRESS=1
