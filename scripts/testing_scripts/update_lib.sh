#!/bin/bash

if [ "$1" = bowman ]; then
  ICPCVER="$(icpc --version | grep icpc | cut -d ' ' -f 3)"
  if [[ "${ICPCVER}" = 17.0.* ]]; then
    module swap gcc/4.7.2 gcc/6.2.0
    module list
  elif [[ "${ICPCVER}" = 17.* || "${ICPCVER}" = 18.* ]]; then
    module swap gcc/4.9.3 gcc/6.2.0
    module list
  fi
   export LIBRARY_PATH=/home/projects/x86-64-knl/gcc/6.2.0/lib/gcc/x86_64-pc-linux-gnu/6.2.0:/home/projects/x86-64-knl/cloog/0.18.4/lib:/home/projects/x86-64-knl/isl/0.16.1/lib:/home/projects/x86-64-knl/gmp/6.1.0/lib:/home/projects/x86-64-knl/mpfr/3.1.3/lib:/home/projects/x86-64-knl/mpc/1.0.3/lib:/home/projects/x86-64-knl/binutils/2.26.0/lib:/usr/lib/gcc/x86_64-redhat-linux/4.8.3:$LIBRARY_PATH
   export LD_LIBRARY_PATH=/home/projects/x86-64-knl/gcc/6.2.0/lib64:/home/projects/x86-64-knl/gcc/6.2.0/lib:/home/projects/x86-64-knl/cloog/0.18.4/lib:/home/projects/x86-64-knl/isl/0.16.1/lib:/home/projects/x86-64-knl/gmp/6.1.0/lib:/home/projects/x86-64-knl/mpfr/3.1.3/lib:/home/projects/x86-64-knl/mpc/1.0.3/lib:/home/projects/x86-64-knl/binutils/2.26.0/lib:/usr/lib/gcc/x86_64-redhat-linux/4.8.3:$LD_LIBRARY_PATH
fi
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
    module swap sems-gcc/4.8.4 sems-gcc/6.4.0
    module list
  fi
fi
