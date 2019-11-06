#!/bin/bash

if [ "$1" = bowman ]; then
   export LIBRARY_PATH=/home/projects/x86-64-knl/gcc/6.2.0/lib/gcc/x86_64-pc-linux-gnu/6.2.0:/home/projects/x86-64-knl/cloog/0.18.4/lib:/home/projects/x86-64-knl/isl/0.16.1/lib:/home/projects/x86-64-knl/gmp/6.1.0/lib:/home/projects/x86-64-knl/mpfr/3.1.3/lib:/home/projects/x86-64-knl/mpc/1.0.3/lib:/home/projects/x86-64-knl/binutils/2.26.0/lib:/usr/lib/gcc/x86_64-redhat-linux/4.8.3:$LIBRARY_PATH
   export LD_LIBRARY_PATH=/home/projects/x86-64-knl/gcc/6.2.0/lib64:/home/projects/x86-64-knl/gcc/6.2.0/lib:/home/projects/x86-64-knl/cloog/0.18.4/lib:/home/projects/x86-64-knl/isl/0.16.1/lib:/home/projects/x86-64-knl/gmp/6.1.0/lib:/home/projects/x86-64-knl/mpfr/3.1.3/lib:/home/projects/x86-64-knl/mpc/1.0.3/lib:/home/projects/x86-64-knl/binutils/2.26.0/lib:/usr/lib/gcc/x86_64-redhat-linux/4.8.3:$LD_LIBRARY_PATH
fi
