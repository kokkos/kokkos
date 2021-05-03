#!/usr/bin/env bash

set -ex

source_dir=${1}
build_dir=${2}
target=${3:-install}

export KOKKOS=${source_dir}
export KOKKOS_BUILD=${build_dir}/kokkos
mkdir -p "$KOKKOS_BUILD"
cd "$KOKKOS_BUILD"
rm -Rf ./*
cmake -G "${CMAKE_GENERATOR:-Ninja}" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
      -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
      -DCMAKE_C_COMPILER="${CC:-cc}" \
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-}" \
      -DKokkos_ENABLE_TESTS="${Kokkos_ENABLE_TESTS:-1}" \
      -DKokkos_ENABLE_"${Kokkos_BACKEND}"=1 \
      -DKokkos_ARCH_"${Kokkos_ARCH}"=1 \
      -DKokkos_ENABLE_HWLOC=On \
      -DKokkos_CXX_STANDARD=14 \
      "$KOKKOS"

time cmake --build . --target "${target}"
