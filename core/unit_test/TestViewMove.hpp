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

#include <gtest/gtest.h>

namespace {

// Check that moving a View outside a parallel region does not increase the
// number of views managing the allocation.
template <class ViewType>
void test_moving_view_does_not_change_use_count(ViewType v) {
  auto* const ptr = v.data();
  auto const cnt  = v.use_count();

  ViewType w(std::move(v));  // move construction
  EXPECT_EQ(w.use_count(), cnt);
  EXPECT_EQ(w.data(), ptr);

  v = std::move(w);  // move assignment
  EXPECT_EQ(v.use_count(), cnt);
  EXPECT_EQ(v.data(), ptr);
}

TEST(TEST_CATEGORY, view_move_and_use_count) {
  using ExecutionSpace = TEST_EXECSPACE;

  test_moving_view_does_not_change_use_count(
      Kokkos::View<int, ExecutionSpace>("v0"));

  test_moving_view_does_not_change_use_count(
      Kokkos::View<float*, ExecutionSpace>("v1", 1));

  Kokkos::View<double**, ExecutionSpace> v2("v2", 1, 2);
  test_moving_view_does_not_change_use_count(
      Kokkos::View<double**, ExecutionSpace>(v2.data(), v2.extent(0),
                                             v2.extent(1)));
  test_moving_view_does_not_change_use_count(
      Kokkos::View<double**, ExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          v2.data(), v2.extent(0), v2.extent(1)));
}

}  // namespace
