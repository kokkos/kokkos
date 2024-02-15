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

#include <TestCuda_Category.hpp>
#include <Test_InterOp_Streams.hpp>
#include <memory>

namespace {

struct StreamsAndDevices {
  std::shared_ptr<std::array<cudaStream_t, 2>> streams;
  std::array<int, 2> devices;

  StreamsAndDevices() {
    int n_devices;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&n_devices));

    devices = {0, n_devices - 1};

    streams = std::shared_ptr<std::array<cudaStream_t, 2>>(
        new std::array<cudaStream_t, 2>{},
        [this](std::array<cudaStream_t, 2> *s) {
          for (int i = 0; i < 2; ++i) {
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[i]));
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy((*s)[i]));
          }
        });
    for (int i = 0; i < 2; ++i) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[i]));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&((*streams)[i])));
    }
  }
};

std::array<TEST_EXECSPACE, 2> get_execution_spaces(
    const StreamsAndDevices &streams_and_devices) {
  TEST_EXECSPACE exec0((*streams_and_devices.streams)[0]);
  TEST_EXECSPACE exec1((*streams_and_devices.streams)[1]);

  // Must return void to use ASSERT_EQ
  [&]() {
    ASSERT_EQ(exec0.cuda_device(), streams_and_devices.devices[0]);
    ASSERT_EQ(exec1.cuda_device(), streams_and_devices.devices[1]);
  }();

  return {exec0, exec1};
}

// Test Interoperability with Cuda Streams
void test_policies(TEST_EXECSPACE exec0, Kokkos::View<int *, TEST_EXECSPACE> v0,
                   TEST_EXECSPACE exec, Kokkos::View<int *, TEST_EXECSPACE> v) {
  using MemorySpace = typename TEST_EXECSPACE::memory_space;

  Kokkos::deep_copy(exec, v, 5);
  Kokkos::deep_copy(exec0, v0, 5);

  Kokkos::deep_copy(v, v0);

  int sum;
  int sum0;

  Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Range_0",
                       Kokkos::RangePolicy<TEST_EXECSPACE>(exec0, 0, 100),
                       Test::FunctorRange<MemorySpace>(v0));
  Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Range",
                       Kokkos::RangePolicy<TEST_EXECSPACE>(exec, 0, 100),
                       Test::FunctorRange<MemorySpace>(v));
  Kokkos::parallel_reduce(
      "Test::cuda::raw_cuda_stream::RangeReduce_0",
      Kokkos::RangePolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(exec0,
                                                                        0, 100),
      Test::FunctorRangeReduce<MemorySpace>(v0), sum0);
  Kokkos::parallel_reduce(
      "Test::cuda::raw_cuda_stream::RangeReduce",
      Kokkos::RangePolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(exec, 0,
                                                                        100),
      Test::FunctorRangeReduce<MemorySpace>(v), sum);
  ASSERT_EQ(600, sum0);
  ASSERT_EQ(600, sum);

  Kokkos::parallel_for("Test::cuda::raw_cuda_stream::MDRange_0",
                       Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                           exec0, {0, 0}, {10, 10}),
                       Test::FunctorMDRange<MemorySpace>(v0));
  Kokkos::parallel_for("Test::cuda::raw_cuda_stream::MDRange",
                       Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                           exec, {0, 0}, {10, 10}),
                       Test::FunctorMDRange<MemorySpace>(v));
  Kokkos::parallel_reduce("Test::cuda::raw_cuda_stream::MDRangeReduce_0",
                          Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                                Kokkos::LaunchBounds<128, 2>>(
                              exec0, {0, 0}, {10, 10}),
                          Test::FunctorMDRangeReduce<MemorySpace>(v0), sum0);
  Kokkos::parallel_reduce("Test::cuda::raw_cuda_stream::MDRangeReduce",
                          Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>,
                                                Kokkos::LaunchBounds<128, 2>>(
                              exec, {0, 0}, {10, 10}),
                          Test::FunctorMDRangeReduce<MemorySpace>(v), sum);
  ASSERT_EQ(700, sum0);
  ASSERT_EQ(700, sum);

  Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Team_0",
                       Kokkos::TeamPolicy<TEST_EXECSPACE>(exec0, 10, 10),
                       Test::FunctorTeam<MemorySpace, TEST_EXECSPACE>(v0));
  Kokkos::parallel_for("Test::cuda::raw_cuda_stream::Team",
                       Kokkos::TeamPolicy<TEST_EXECSPACE>(exec, 10, 10),
                       Test::FunctorTeam<MemorySpace, TEST_EXECSPACE>(v));
  Kokkos::parallel_reduce(
      "Test::cuda::raw_cuda_stream::Team_0",
      Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(exec0,
                                                                       10, 10),
      Test::FunctorTeamReduce<MemorySpace, TEST_EXECSPACE>(v0), sum0);
  Kokkos::parallel_reduce(
      "Test::cuda::raw_cuda_stream::Team",
      Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::LaunchBounds<128, 2>>(exec, 10,
                                                                       10),
      Test::FunctorTeamReduce<MemorySpace, TEST_EXECSPACE>(v), sum);
  ASSERT_EQ(800, sum0);
  ASSERT_EQ(800, sum);
}

TEST(cuda_multi_gpu, managed_views) {
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

TEST(cuda_multi_gpu, unmanaged_views) {
  StreamsAndDevices streams_and_devices;
  {
    std::array<TEST_EXECSPACE, 2> execs =
        get_execution_spaces(streams_and_devices);

    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(execs[0].cuda_device()));
    int *p0;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void **>(&p0), sizeof(int) * 100));
    Kokkos::View<int *, TEST_EXECSPACE> view0(p0, 100);

    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(execs[1].cuda_device()));
    int *p;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void **>(&p), sizeof(int) * 100));
    Kokkos::View<int *, TEST_EXECSPACE> view(p, 100);

    test_policies(execs[0], view0, execs[1], view);
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p0));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p));
  }
}
}  // namespace
