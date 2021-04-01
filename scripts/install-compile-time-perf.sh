#!/bin/bash

# usage: 
#
#   ./install-compile-time-perf.sh -DCMAKE_INSTALL_PREFIX=/usr/local
#

EXISTING=$(which compile-time-perf-analyzer)

set +e

if [ -n "${EXISTING}" ]; then 
    echo "compile-time-perf already installed in $(dirname ${EXISTING})"
    exit 0
fi

WORKING_DIR=$(mktemp -d -t install-ctp)

if [ ! -d "${WORKING_DIR}" ]; then
    mkdir -p ${WORKING_DIR}
fi

cd ${WORKING_DIR}
git clone https://github.com/jrmadsen/compile-time-perf.git
cmake -B build-ctp $@ compile-time-perf
cmake --build build-ctp --target all
cmake --build build-ctp --target install
