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
#include <type_traits>

#define KOKKOS_IMPL_TEST_ACCESS

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN

template <class T, class ExecutionSpace>
struct TestViewMDSpanConversion {
  using value_type = T;

  template <class MDSpanLayout, class KokkosLayout, class DataType,
            class MDSpanExtents>
  static void test_conversion_from_mdspan(Kokkos::View<DataType> ref,
                                          const MDSpanExtents &exts) {
    using view_type   = Kokkos::View<DataType, KokkosLayout,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using natural_mdspan_type = typename Kokkos::Experimental::Impl::MDSpanViewTraits<typename view_type::traits>::mdspan_type;
    using mdspan_type = Kokkos::mdspan<value_type, MDSpanExtents, MDSpanLayout>;

    static_assert(std::is_convertible_v<mdspan_type, natural_mdspan_type>);

    // Manually create an mdspan from ref so we have a valid pointer to play
    // with
    auto mds = mdspan_type{ref.data(), exts};

    auto test_view = view_type(mds);

    ASSERT_EQ(test_view.data(), ref.data());
    ASSERT_EQ(test_view.data(), mds.data_handle());
    for (std::size_t r = 0; r < mdspan_type::rank(); ++r) {
      ASSERT_EQ(test_view.extent(r), ref.extent(r));
      ASSERT_EQ(test_view.extent(r), exts.extent(r));
    }

    natural_mdspan_type cvt = test_view;
  }

  static void run_test() {
    static_assert(std::is_same_v<
                  typename Kokkos::Experimental::Impl::ArrayLayoutFromLayout<
                      Kokkos::Experimental::layout_left_padded<sizeof(
                          value_type)>>::type,
                  Kokkos::LayoutLeft>);
    static_assert(std::is_same_v<
                  typename Kokkos::Experimental::Impl::ArrayLayoutFromLayout<
                      Kokkos::Experimental::layout_right_padded<sizeof(
                          value_type)>>::type,
                  Kokkos::LayoutRight>);

    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(
        Kokkos::View<double *>("ref", 7),
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(7));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<double[7]>("ref"),
                            Kokkos::extents<std::size_t, 7>());
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(
        Kokkos::View<double[7]>("ref"),
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(7));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<double *>("ref", 7),
                            Kokkos::extents<std::size_t, 7>());

    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<double **>("ref", 7, 3),
                            Kokkos::dextents<std::size_t, 2>(7, 3));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<double[7][3]>("ref"),
                            Kokkos::extents<std::size_t, 7, 3>());
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<double[7][3]>("ref"),
                            Kokkos::extents<std::size_t, Kokkos::dynamic_extent,
                                            Kokkos::dynamic_extent>(7, 3));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<double **>("ref", 7, 3),
                            Kokkos::extents<std::size_t, 7, 3>());
  }
};

namespace Test {

TEST(TEST_CATEGORY, view_mdspan_conversion) {
  TestViewMDSpanConversion<double, TEST_EXECSPACE>::run_test();
}

}  // namespace Test

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN
