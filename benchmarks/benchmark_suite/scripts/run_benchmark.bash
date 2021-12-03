#!/bin/bash
readonly SCRIPT_PATH=$1
readonly KOKKOS_DEVICES=$2
readonly KOKKOS_ARCH=$3
readonly COMPILER=$4
if [[ $# -lt 4 ]]; then
  echo "Usage: ./run_benchmark.bash PATH_TO_SCRIPTS KOKKOS_DEVICES KOKKOS_ARCH COMPILER"; exit 1
fi

function get_logfile() {
  local logfile=run-benchmark--$(date +"%Y-%m-%d-%H-%M").log

  cd ${PWD}/kokkos
  local git_info="KOKKOS_BRANCH: $(git branch --show-current)
KOKKOS_SHA: $(git rev-parse --short HEAD)"
  cd ..

  echo "KOKKOS_DEVICES: $KOKKOS_DEVICES
KOKKOS_ARCH: $KOKKOS_ARCH
$git_info
COMPILER: $($COMPILER --version | head -n 1)
" >> $logfile

  echo $logfile
}

${SCRIPT_PATH}/checkout_repos.bash
${SCRIPT_PATH}/build_code.bash --arch=${KOKKOS_ARCH} --device-list=${KOKKOS_DEVICES} --compiler=${COMPILER}
logfile=$(get_logfile)
for _ in {1..1}  # increase the range to run benchmark multiple times
do
  ${SCRIPT_PATH}/run_tests.bash | tee -a $logfile
done
