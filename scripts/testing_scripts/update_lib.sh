#!/bin/bash

if [ "$1" = blake ]; then
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.* || "${ICPCVER}" = 18.0.128 ]]; then
    module swap gcc/4.9.3 gcc/6.4.0
    module list
  fi
fi
if [ "$1" = kokkos-dev ]; then
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.* ]]; then
    module swap sems-gcc/4.9.3 sems-gcc/6.4.0
    module list
  fi
fi
if [ "$1" = kokkos-dev-2 ]; then
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.* ]]; then
    module swap sems-gcc/4.9.3 sems-gcc/6.4.0
    module list
  fi
fi
if [ "$1" = sems ]; then
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.* ]]; then
    module swap sems-gcc/4.9.3 sems-gcc/6.4.0
    module list
  fi
fi
