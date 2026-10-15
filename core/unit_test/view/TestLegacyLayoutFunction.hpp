// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

template <class ViewT, class Exts>
void test_layout_single_rank_impl(ViewT a, Exts exts, int stride) {
  // Avoid warning for pointless comparison of unsigned with 0
  if constexpr (ViewT::rank() > 0) {
    for (unsigned r = 0; r < ViewT::rank(); r++) {
      ASSERT_EQ(a.extent(r), exts.extent(r));
      ASSERT_EQ(a.layout().dimension[r], exts.extent(r));
    }
  }
  if constexpr (ViewT::rank() > 1) {
    // In Legacy View the stride wasn't propagated correctly
    ASSERT_EQ(a.layout().stride, stride);
    int actual_stride =
        std::is_same_v<typename ViewT::array_layout, Kokkos::LayoutLeft>
            ? a.stride(1)
            : a.stride(ViewT::rank() - 2);
    ASSERT_EQ(actual_stride, stride);
  }
  {
    ViewT b("B", a.layout());
    ASSERT_EQ(a.layout(), b.layout());
    ASSERT_EQ(a.mapping(), b.mapping());
  }
  {
    ViewT b(a.data(), a.layout());
    ASSERT_EQ(a.layout(), b.layout());
    ASSERT_EQ(a.mapping(), b.mapping());
  }
  {
    ViewT b(Kokkos::view_alloc("B"), a.layout());
    ASSERT_EQ(a.layout(), b.layout());
    ASSERT_EQ(a.mapping(), b.mapping());
  }
  {
    ViewT b(Kokkos::view_wrap(a.data()), a.layout());
    ASSERT_EQ(a.layout(), b.layout());
    ASSERT_EQ(a.mapping(), b.mapping());
  }
}

template <class ViewT, std::integral... Sizes>
void test_layout_single_rank(Sizes... sizes) {
  using extents_type = typename ViewT::extents_type;
  using mapping_type = typename ViewT::mapping_type;

  extents_type exts{sizes...};

  int padded_extent = 1;

  if (ViewT::rank() > 0) {
    padded_extent =
        std::is_same_v<typename ViewT::array_layout, Kokkos::LayoutLeft>
            ? exts.extent(0)
            : exts.extent(ViewT::rank() - 1);
  }
  // Test with no padding
  test_layout_single_rank_impl(ViewT("A", sizes...), exts, padded_extent);
  // using array_layout because I need LayoutLeft/Right, not
  // layout_left/right_padded
  using Layout = typename ViewT::array_layout;
  Layout l;
  // Avoid warning for pointless comparison with 0
  if constexpr (ViewT::rank() > 0)
    for (size_t r = 0; r < ViewT::rank(); r++) l.dimension[r] = exts.extent(r);
  test_layout_single_rank_impl(ViewT("A", l), exts, padded_extent);
  test_layout_single_rank_impl(ViewT("A", mapping_type(exts)), exts,
                               padded_extent);

  // Test with padding
  padded_extent += 3;

  l.stride = padded_extent;
  test_layout_single_rank_impl(ViewT("A", l), exts, padded_extent);

  test_layout_single_rank_impl(ViewT("A", mapping_type(exts, padded_extent)),
                               exts, padded_extent);
}

template <class Layout>
void test_layout() {
  test_layout_single_rank<Kokkos::View<double, Layout, TEST_EXECSPACE>>();
  test_layout_single_rank<Kokkos::View<double*, Layout, TEST_EXECSPACE>>(7);
  test_layout_single_rank<Kokkos::View<double[7], Layout, TEST_EXECSPACE>>(7);

  test_layout_single_rank<Kokkos::View<double**, Layout, TEST_EXECSPACE>>(7,
                                                                          11);
  test_layout_single_rank<Kokkos::View<double* [11], Layout, TEST_EXECSPACE>>(
      7, 11);
  test_layout_single_rank<Kokkos::View<double*******, Layout, TEST_EXECSPACE>>(
      7, 11, 13, 17, 19, 2, 3);
  test_layout_single_rank<
      Kokkos::View<double***** [2][3], Layout, TEST_EXECSPACE>>(7, 11, 13, 17,
                                                                19, 2, 3);
}

TEST(TEST_CATEGORY, view_legacy_layout_member) {
  test_layout<Kokkos::LayoutLeft>();
  test_layout<Kokkos::LayoutRight>();
}
