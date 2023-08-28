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

#include <Kokkos_Core.hpp>
#include <TestCuda_Category.hpp>
#include <gtest/gtest.h>

namespace Test {
// Test Interoperability with Cuda Streams and multiple GPUs.
TEST(cuda, raw_cuda_streams) {
  Kokkos::ScopeGuard scope_guard;

  cudaStream_t stream;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  {
    TEST_EXECSPACE cuda_instance(TEST_EXECSPACE().cuda_device(), stream);
    ASSERT_EQ(cuda_instance.cuda_device(), TEST_EXECSPACE().cuda_device());
  }
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
}
}  // namespace Test
