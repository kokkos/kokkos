// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_DynRankView.hpp>

namespace {

template <class Layout>
void test_dyn_rank_view_layout_member() {
  {
    Kokkos::DynRankView<int, Layout> a(
        Kokkos::View<int***, Layout>("A", 11, 7, 5));
    auto l = a.layout();
    ASSERT_EQ(l.dimension[0], 11lu);
    ASSERT_EQ(l.dimension[1], 7lu);
    ASSERT_EQ(l.dimension[2], 5lu);
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
    ASSERT_EQ(l.stride, KOKKOS_INVALID_INDEX);
#else
    ASSERT_EQ(l.stride, (std::is_same_v<Layout, Kokkos::LayoutLeft>
                             ? 11lu
                             : KOKKOS_INVALID_INDEX));
#endif
  }
  {
    Kokkos::DynRankView<int, Layout> a(Kokkos::View<int**, Layout>("A", 7, 5));
    auto l = a.layout();
    ASSERT_EQ(l.dimension[0], 7lu);
    ASSERT_EQ(l.dimension[1], 5lu);
#ifdef KOKKOS_ENABLE_IMPL_VIEW_LEGACY
    ASSERT_EQ(l.stride, KOKKOS_INVALID_INDEX);
#else
    ASSERT_EQ(l.stride,
              (std::is_same_v<Layout, Kokkos::LayoutLeft> ? 7lu : 5lu));
#endif
  }
}

}  // namespace

TEST(TEST_CATEGORY, dyn_rank_view_layout_member) {
  test_dyn_rank_view_layout_member<Kokkos::LayoutRight>();
  test_dyn_rank_view_layout_member<Kokkos::LayoutLeft>();
}
