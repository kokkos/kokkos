#!/usr/bin/env bash

set -exo pipefail

source_dir=${1}
build_dir=${2}

export KOKKOS=${source_dir}
export KOKKOS_BUILD=${build_dir}/kokkos
pushd "$KOKKOS_BUILD"

ctest --output-on-failure | tee cmake-output.log

popd
