#!/bin/bash

local machine_input="$1"
local compiler_input="$2"

check_sems_intel() {
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.* ]]; then
    module swap sems-gcc/4.9.3 sems-gcc/6.4.0
    module list
  fi
  if [[ "${ICPCVER}" = 19.* ]]; then
    # Newer gcc needed for c++ standard beyond c++14
    module swap sems-gcc/6.1.0 sems-gcc/7.2.0
    module list
  fi
}

check_sems_clang() {
  CLANGVER=$(clang --version | grep "clang version" | cut -d " " -f 3)
  if [[ "${CLANGVER}" = 9.* ]] || [[ "${CLANGVER}" = 10.* ]]; then
    # Newer gcc needed for c++ standard beyond c++14
    module swap sems-gcc/5.3.0 sems-gcc/8.3.0
    module list
  fi
}

check_compiler_modules() {
  if [[ "$compiler_input" = clang/* ]]; then
    echo "  clang compiler - check supporting modules"
    check_sems_clang
  elif [[ "$compiler_input" = intel/* ]]; then
    echo "  intel compiler - check supporting modules"
    check_sems_intel
  fi
}

if [ "$machine_input" = blake ]; then
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.* || "${ICPCVER}" = 18.0.128 ]]; then
    module swap gcc/4.9.3 gcc/6.4.0
    module list
  fi
fi
if [ "$machine_input" = kokkos-dev ]; then
  check_compiler_modules
fi
if [ "$machine_input" = kokkos-dev-2 ]; then
  check_compiler_modules
fi
if [ "$machine_input" = sems ] || [ "$machine_input" = sogpu ]; then
  check_compiler_modules
fi
