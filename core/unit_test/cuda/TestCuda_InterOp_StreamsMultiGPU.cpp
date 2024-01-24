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

namespace {
TEST(cuda, multi_gpu) {
  Kokkos::initialize();

  int n_devices;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&n_devices));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(0));
  cudaStream_t stream0;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream0));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(n_devices - 1));
  cudaStream_t stream;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));

  {
    TEST_EXECSPACE space0(stream0);
    ASSERT_EQ(space0.cuda_device(), 0);
    TEST_EXECSPACE space(stream);
    ASSERT_EQ(space.cuda_device(), n_devices - 1);
  }
  Kokkos::finalize();

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(0));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream0));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(n_devices - 1));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
}
}  // namespace
