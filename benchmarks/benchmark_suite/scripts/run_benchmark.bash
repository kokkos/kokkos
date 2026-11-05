#!/bin/bash

# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

SCRIPT_PATH=$1
KOKKOS_DEVICES=$2
KOKKOS_ARCH=$3
COMPILER=$4
if [[ $# < 4 ]]; then
  echo "Usage: ./run_benchmark.bash PATH_TO_SCRIPTS KOKKOS_DEVICES KOKKOS_ARCH COMPILER"
else

${SCRIPT_PATH}/checkout_repos.bash
${SCRIPT_PATH}/build_code.bash --arch=${KOKKOS_ARCH} --device-list=${KOKKOS_DEVICES} --compiler=${COMPILER}
${SCRIPT_PATH}/run_tests.bash

fi
