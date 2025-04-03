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
#include <Kokkos_Core.hpp>

namespace {

TEST(cuda, unified_memory_assign) {
#if !defined(KOKKOS_ENABLE_IMPL_CUDA_UNIFIED_MEMORY)
  GTEST_SKIP()
      << "this test should only be run with CUDA unified memory enabled";
#else
  Kokkos::View<int*, Kokkos::HostSpace> view_host("view_host", 10);
  Kokkos::View<int*, Kokkos::CudaHostPinnedSpace> view_hostpinned = view_host;
  Kokkos::View<int*, Kokkos::CudaUVMSpace> view_shared            = view_host;
  Kokkos::View<int*, Kokkos::CudaSpace> view_device               = view_host;

  EXPECT_EQ(view_host.use_count(), 4);
  EXPECT_EQ(view_hostpinned.use_count(), 4);
  EXPECT_EQ(view_shared.use_count(), 4);
  EXPECT_EQ(view_device.use_count(), 4);

  EXPECT_EQ(view_host.data(), view_hostpinned.data());
  EXPECT_EQ(view_host.data(), view_shared.data());
  EXPECT_EQ(view_host.data(), view_device.data());
#endif
}
}  // namespace
