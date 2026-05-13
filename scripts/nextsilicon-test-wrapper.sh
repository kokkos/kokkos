#! /bin/bash

# This script has two different modes of operation:
# if KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS is anything, we run in a telemetry-less mode, where any projection error counts as a failure
# else, we run the two-pass telemetry-based flow

set -e
set -x

# make sure nextsystemd shuts down no matter what
cleanup() {
        # code from last thing that ran
        local result=$?
        echo "cleanup..."
        nextcli system terminate || true
        wait ${NEXTSYSTEMD_PID}
        echo "exit with $result"
        exit $result
}
trap cleanup EXIT

patch_dir=$(mktemp -d)
if [[ -n "${KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS:-}" ]]; then
    echo "nextsilicon-test-wrapper.sh: KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS is set"

    # configure nextsystemd for telemetry-less mode
    echo -e "optimizer-pi:\n  enable-telemetry-less: true" > ${patch_dir}/kokkos.patch

    # start nextsystemd without profiling tools data collection
    nextsystemd --ui-collector-address none --cfg-file ${patch_dir}/kokkos.patch &
    NEXTSYSTEMD_PID=$!

    # telemetry-less run
    ./"$1" "${@:2}"
else
    echo "nextsilicon-test-wrapper.sh: KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS is not set"

    # configure nextsystemd to try to offload every parallel region
    echo -e "optimizer-pi:\n  mlc:\n    acceleration-threshold: 1" > ${patch_dir}/kokkos.patch

    # start nextsystemd without profiling tools data collection
    nextsystemd --ui-collector-address none --cfg-file ${patch_dir}/kokkos.patch &
    NEXTSYSTEMD_PID=$!

    # training run
    ./"$1" "${@:2}"

   # wait for optimization/projection to finish, up to 5 minutes
   SECONDS=0
   while [ $SECONDS -lt 300 ]; do
       # check current state
       ret=0
       status="$(nextcli application status | grep 'Optimization state:')"
       if [[ $status == *IDLE* ]]; then
           # no mills found, return
           exit 0
       elif [[ $status == *IMPROVED* ]]; then
           # optimization/projection finished, do device run
           ./"$1" "${@:2}"
           exit
       elif [[ $status == *OPTIMIZING* ]]; then
           # optimization/projection still running, wait 10 more seconds
           sleep 10
           continue
       else
           # in some other state, something went wrong
           exit 2
       fi
   done

   echo "error: timed out" >&2
   exit 127
fi
