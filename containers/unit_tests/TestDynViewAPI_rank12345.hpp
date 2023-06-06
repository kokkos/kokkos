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

#include <TestDynViewAPI.hpp>

namespace Test {
TEST(TEST_CATEGORY, dyn_rank_view_api_operator_rank12345) {
  TestDynViewAPI<double, TEST_EXECSPACE>::run_operator_test_rank12345();
}

template <typename SharedMemorySpace>
void test_dyn_rank_view_resize() {
  int n = 1000000;
  Kokkos::DynRankView<double, SharedMemorySpace> device_view("device view", n);
  auto device_view_copy = device_view;

  Kokkos::resize(device_view, 2 * n);

  for (int i = 0; i < 2 * n; ++i) device_view(i) = i + 1;

  Kokkos::fence();

  for (int i = 0; i < 2 * n; ++i) ASSERT_EQ(device_view(i), i + 1);
}

template <typename SharedMemorySpace>
void test_dyn_rank_view_realloc() {
  int n = 1000000;
  Kokkos::DynRankView<double, SharedMemorySpace> device_view("device view", n);
  auto device_view_copy = device_view;

  Kokkos::realloc(device_view, 2 * n);

  for (int i = 0; i < 2 * n; ++i) device_view(i) = i + 1;

  Kokkos::fence();

  for (int i = 0; i < 2 * n; ++i) ASSERT_EQ(device_view(i), i + 1);
}

TEST(TEST_CATEGORY, dyn_rank_view_resize_realloc) {
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>) {
    test_dyn_rank_view_resize<Kokkos::SharedSpace>();
    test_dyn_rank_view_realloc<Kokkos::SharedSpace>();
  } else
    GTEST_SKIP() << "skipping since we only test SharedSpace";
}

}  // namespace Test
