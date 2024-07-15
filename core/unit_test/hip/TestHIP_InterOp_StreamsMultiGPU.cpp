//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <TestHIP_Category.hpp>
#include <TestMultiGPU.hpp>

namespace {

struct StreamsAndDevices {
  std::array<hipStream_t, 2> streams;
  std::array<int, 2> devices;

  StreamsAndDevices() {
    int n_devices;
    KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceCount(&n_devices));

    devices = {0, n_devices - 1};
    for (int i = 0; i < 2; ++i) {
      KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(devices[i]));
      KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamCreate(&streams[i]));
    }
  }
  StreamsAndDevices(const StreamsAndDevices &) = delete;
  StreamsAndDevices &operator=(const StreamsAndDevices &) = delete;
  ~StreamsAndDevices() {
    for (int i = 0; i < 2; ++i) {
      KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(devices[i]));
      KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamDestroy(streams[i]));
    }
  }
};

std::array<TEST_EXECSPACE, 2> get_execution_spaces(
    const StreamsAndDevices &streams_and_devices) {
  TEST_EXECSPACE exec0(streams_and_devices.streams[0]);
  TEST_EXECSPACE exec1(streams_and_devices.streams[1]);

  // Must return void to use ASSERT_EQ
  [&]() {
    ASSERT_EQ(exec0.hip_device(), streams_and_devices.devices[0]);
    ASSERT_EQ(exec1.hip_device(), streams_and_devices.devices[1]);
  }();

  return {exec0, exec1};
}

TEST(hip_multi_gpu, managed_views) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    Kokkos::View<int *, TEST_EXECSPACE> view0(
        Kokkos::view_alloc("v0", execs[0]), 100);
    Kokkos::View<int *, TEST_EXECSPACE> view(Kokkos::view_alloc("v", execs[1]),
                                             100);

    test_policies(execs[0], view0, execs[1], view);
  }
}

TEST(hip_multi_gpu, unmanaged_views) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(execs[0].hip_device()));
    int *p0;
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMalloc(reinterpret_cast<void **>(&p0), sizeof(int) * 100));
    Kokkos::View<int *, TEST_EXECSPACE> view0(p0, 100);

    KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(execs[1].hip_device()));
    int *p;
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMalloc(reinterpret_cast<void **>(&p), sizeof(int) * 100));
    Kokkos::View<int *, TEST_EXECSPACE> view(p, 100);

    test_policies(execs[0], view0, execs[1], view);
    KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(p0));
    KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(p));
  }
}

TEST(hip_multi_gpu, scratch_space) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    test_scratch(execs[0], execs[1]);
  }
}
}  // namespace
