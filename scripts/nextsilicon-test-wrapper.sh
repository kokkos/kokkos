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

    # if there are mills, we will get to optimized
    # if there are no mills, we'll get to idle really fast

    # if there are no mills, we will get to idle really fast
    status="$(nextcli application wait --timeout 10 2>&1 || true)"

    if [[ $status != *IDLE* ]]; then
        # If there are mills (status is not IDLE), then we will eventually get to OPTIMIZED (or error)
        nextcli application wait --timeout 300
        # handoff run
        ./"$1" "${@:2}"
    fi

fi
