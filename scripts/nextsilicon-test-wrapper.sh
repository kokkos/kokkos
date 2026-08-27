#! /bin/bash

# This script has two different modes of operation:
# if KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS is anything, we run in a telemetry-less mode, where any projection error counts as a failure
# else, we run the two-pass telemetry-based flow

set -e

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

# Kill any lingering NS processes left by a previously timed-out test
pkill -x nextsystemd   2>/dev/null || true
pkill -x nextoptd      2>/dev/null || true
pkill -x nextruntimed  2>/dev/null || true
pkill -f "webapps/collector/influxdb" 2>/dev/null || true
rm -f "/run/user/$(id -u)/nextsystemd/observability"/*.sock 2>/dev/null || true
sleep 2

if [[ -n "${KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS:-}" ]]; then
    echo "nextsilicon-test-wrapper.sh: KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS is set"

    # configure nextsystemd
    tee ${patch_dir}/kokkos.patch <<EOF
optimizer-pi:
  # switch to telemetry-less mode
  enable-telemetry-less: true
projection:
  # increase cache size to avoid eviction of projection data
  cache-max-size-mib: 8192
EOF

    # start nextsystemd without profiling tools data collection
    nextsystemd --ui-collector-address none --cfg-file ${patch_dir}/kokkos.patch &
    NEXTSYSTEMD_PID=$!

    # verify that nextsystemd is running
    nextcli system status | grep -q 'Service: UP' || exit 127

    # telemetry-less run
    nextloader -- ./"$1" "${@:2}"
else
    echo "nextsilicon-test-wrapper.sh: KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS is not set"

    # configure nextsystemd
    tee ${patch_dir}/kokkos.patch <<EOF
optimizer-pi:
  mlc:
    # try to offload every parallel region
    acceleration-threshold: 1
projection:
  # increase cache size to avoid eviction of projection data
  cache-max-size-mib: 8192
EOF

    # start nextsystemd without profiling tools data collection
    nextsystemd --ui-collector-address none --cfg-file ${patch_dir}/kokkos.patch &
    NEXTSYSTEMD_PID=$!

    # verify that nextsystemd is running
    nextcli system status | grep -q 'Service: UP' || exit 127

    # training run
    nextloader -- ./"$1" "${@:2}"

    # wait for optimization to start
    sleep 15

    # wait for optimization/projection to finish, up to 1.5 hours
    SECONDS=0
    while [ $SECONDS -lt 5400 ]; do
        # check current state
        ret=0
        status="$(nextcli application status | grep 'Optimization state:')"
        if [[ $status == *IDLE* ]]; then
            # no mills found, return
            exit 0
        elif [[ $status == *IMPROVED* ]]; then
            # optimization/projection finished, do device run
            nextloader -- ./"$1" "${@:2}"
            exit
        elif [[ $status == *OPTIMIZING* ]]; then
            # optimization/projection still running, wait 30 more seconds
            sleep 30
            continue
        else
            # in some other state, something went wrong
            exit 2
        fi
    done

    echo "error: timed out" >&2
    exit 127
fi
