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

#include <gtest/gtest.h>

#include <Kokkos_DynRankView.hpp>

namespace {

template <class Layout>
void test_dyn_rank_view_layout_member() {
  bool is_ll = std::is_same_v<Layout, Kokkos::LayoutLeft>;
  {
    Kokkos::DynRankView<int, Layout> a(
        Kokkos::View<int***, Layout>("A", 11, 7, 5));
    auto l = a.layout();
    ASSERT_EQ(l.dimension[0], 11lu);
    ASSERT_EQ(l.dimension[1], 7lu);
    ASSERT_EQ(l.dimension[2], 5lu);
    ASSERT_TRUE(
        (l.stride == is_ll ? 11lu : 5lu || l.stride == KOKKOS_INVALID_INDEX));
  }
  {
    Kokkos::DynRankView<int, Layout> a(Kokkos::View<int**, Layout>("A", 7, 5));
    auto l = a.layout();
    ASSERT_EQ(l.dimension[0], 7lu);
    ASSERT_EQ(l.dimension[1], 5lu);
    ASSERT_TRUE(
        (l.stride == is_ll ? 7lu : 5lu || l.stride == KOKKOS_INVALID_INDEX));
  }
}

}  // namespace

TEST(TEST_CATEGORY, dyn_rank_view_layout_member) {
  test_dyn_rank_view_layout_member<Kokkos::LayoutRight>();
  test_dyn_rank_view_layout_member<Kokkos::LayoutLeft>();
}
