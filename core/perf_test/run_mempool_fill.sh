#!/bin/bash -e
NT=$1
PROG="./KokkosCore_PerformanceTest_Mempool"
COMMON_ARGS="--kokkos-threads=$NT --fill_stride=1 --alloc_size=10000000 --super_size=1000000 --chunk_span=5 --repeat_inner=100"

postproc() {
cat log | tail -n 1 | rev | cut -d ' ' -f 1 | rev >> yvals
}

for yset in 1
do
  rm -f xvals yvals
  for x in 20 40 60 80 100 
  do
    echo "yset $yset x fill $x"
    echo $x >> xvals
    $PROG $COMMON_ARGS --fill_level=$x > log
    postproc
  done
  rm -f yvals$yset
  mv yvals yvals$yset
done

rm -f datapoints_fill
paste -d',' xvals yvals1 > datapoints_fill
