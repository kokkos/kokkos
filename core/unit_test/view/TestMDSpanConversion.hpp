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

#include <Kokkos_Core.hpp>
#include "experimental/__p0009_bits/default_accessor.hpp"
#include "experimental/__p0009_bits/dynamic_extent.hpp"
#include "experimental/__p2642_bits/layout_padded_fwd.hpp"

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN

template <class T, class ExecutionSpace>
struct TestViewMDSpanConversion {
  using value_type = T;

  struct test_accessor
  {
    using offset_policy = test_accessor;
    using element_type = value_type;
    using reference = element_type &;
    using data_handle_type = element_type *;

    constexpr test_accessor() noexcept = default;
    constexpr reference access(data_handle_type p, std::size_t i) {
      return p[i];
    }
    constexpr data_handle_type offset(data_handle_type p, std::size_t i) {
      return p + i;
    }
  };

  template <class MDSpanLayout, class KokkosLayout, class DataType,
            class MDSpanExtents, class... RefViewProps>
  static void test_conversion_from_mdspan(Kokkos::View<DataType, RefViewProps...> ref,
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
    ASSERT_EQ(test_view.layout(), ref.layout());
    for (std::size_t r = 0; r < mdspan_type::rank(); ++r) {
      ASSERT_EQ(test_view.extent(r), ref.extent(r));
      ASSERT_EQ(test_view.extent(r), exts.extent(r));
    }
  }

  template <class MDSpanLayoutMapping, class ViewType>
  static void test_conversion_to_mdspan(
      const MDSpanLayoutMapping &ref_layout_mapping, ViewType v) {
    using view_type = ViewType;
    using natural_mdspan_type =
        typename Kokkos::Experimental::Impl::MDSpanViewTraits<
            typename view_type::traits>::mdspan_type;

    natural_mdspan_type cvt = v;
    ASSERT_EQ(cvt.data_handle(), v.data());
    ASSERT_EQ(cvt.mapping(), ref_layout_mapping);
  }

  template <class MDSpanLayoutMapping, class ViewType, class AccessorType>
  static void test_conversion_to_mdspan(
      const MDSpanLayoutMapping &ref_layout_mapping, ViewType v,
      const AccessorType &a) {
    using view_type = ViewType;
    using natural_mdspan_type =
        typename Kokkos::Experimental::Impl::MDSpanViewTraits<
            typename view_type::traits>::mdspan_type;

    auto cvt = v.to_mdspan(a);
    ASSERT_EQ(cvt.data_handle(), v.data());
    ASSERT_EQ(cvt.mapping(), ref_layout_mapping);
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

    // LayoutLeft
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(
        Kokkos::View<value_type *, Kokkos::LayoutLeft>("ref", 7),
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(7));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<value_type[7], Kokkos::LayoutLeft>("ref"),
                            Kokkos::extents<std::size_t, 7>());
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(
        Kokkos::View<value_type[7], Kokkos::LayoutLeft>("ref"),
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(7));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<value_type *, Kokkos::LayoutLeft>("ref", 7),
                            Kokkos::extents<std::size_t, 7>());

    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<value_type **, Kokkos::LayoutLeft>("ref", 7, 3),
                            Kokkos::dextents<std::size_t, 2>(7, 3));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<value_type[7][3], Kokkos::LayoutLeft>("ref"),
                            Kokkos::extents<std::size_t, 7, 3>());
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<value_type[7][3], Kokkos::LayoutLeft>("ref"),
                            Kokkos::extents<std::size_t, Kokkos::dynamic_extent,
                                            Kokkos::dynamic_extent>(7, 3));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_left_padded<sizeof(value_type)>,
        Kokkos::LayoutLeft>(Kokkos::View<value_type **, Kokkos::LayoutLeft>("ref", 7, 3),
                            Kokkos::extents<std::size_t, 7, 3>());

    // LayoutRight
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(
        Kokkos::View<value_type *, Kokkos::LayoutRight>("ref", 7),
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(7));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(Kokkos::View<value_type[7], Kokkos::LayoutRight>("ref"),
                            Kokkos::extents<std::size_t, 7>());
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(
        Kokkos::View<value_type[7], Kokkos::LayoutRight>("ref"),
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(7));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(Kokkos::View<value_type *, Kokkos::LayoutRight>("ref", 7),
                            Kokkos::extents<std::size_t, 7>());

    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(Kokkos::View<value_type **, Kokkos::LayoutRight>("ref", 7, 3),
                            Kokkos::dextents<std::size_t, 2>(7, 3));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(Kokkos::View<value_type[7][3], Kokkos::LayoutRight>("ref"),
                            Kokkos::extents<std::size_t, 7, 3>());
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(Kokkos::View<value_type[7][3], Kokkos::LayoutRight>("ref"),
                            Kokkos::extents<std::size_t, Kokkos::dynamic_extent,
                                            Kokkos::dynamic_extent>(7, 3));
    test_conversion_from_mdspan<
        Kokkos::Experimental::layout_right_padded<sizeof(value_type)>,
        Kokkos::LayoutRight>(Kokkos::View<value_type **, Kokkos::LayoutRight>("ref", 7, 3),
                            Kokkos::extents<std::size_t, 7, 3>());

    // Conversion to mdspan
    using layout_left_padded = Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>;
    using layout_right_padded = Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>;
    test_conversion_to_mdspan(layout_left_padded::mapping<Kokkos::extents<std::size_t, 4>>({}, 4), Kokkos::View<value_type *, Kokkos::LayoutLeft>("v", 4));
    test_conversion_to_mdspan(layout_left_padded::mapping<Kokkos::extents<std::size_t, 4, 7>>({}, 4), Kokkos::View<value_type **, Kokkos::LayoutLeft>("v", 4, 7));

    test_conversion_to_mdspan(layout_right_padded::mapping<Kokkos::extents<std::size_t, 4>>({}, 4), Kokkos::View<value_type *, Kokkos::LayoutRight>("v", 4));
    test_conversion_to_mdspan(layout_right_padded::mapping<Kokkos::extents<std::size_t, 4, 7>>({}, 7), Kokkos::View<value_type **, Kokkos::LayoutRight>("v", 4, 7));

    test_conversion_to_mdspan(layout_left_padded::mapping<Kokkos::extents<std::size_t, 4>>({}, 4), Kokkos::View<value_type *, Kokkos::LayoutLeft>("v", 4), Kokkos::default_accessor<value_type>{});
    test_conversion_to_mdspan(layout_left_padded::mapping<Kokkos::extents<std::size_t, 4, 7>>({}, 4), Kokkos::View<value_type **, Kokkos::LayoutLeft>("v", 4, 7), Kokkos::default_accessor<value_type>{});

    test_conversion_to_mdspan(layout_right_padded::mapping<Kokkos::extents<std::size_t, 4>>({}, 4), Kokkos::View<value_type *, Kokkos::LayoutRight>("v", 4), Kokkos::default_accessor<value_type>{});
    test_conversion_to_mdspan(layout_right_padded::mapping<Kokkos::extents<std::size_t, 4, 7>>({}, 7), Kokkos::View<value_type **, Kokkos::LayoutRight>("v", 4, 7), Kokkos::default_accessor<value_type>{});

    test_conversion_to_mdspan(layout_left_padded::mapping<Kokkos::extents<std::size_t, 4>>({}, 4), Kokkos::View<value_type *, Kokkos::LayoutLeft>("v", 4), test_accessor{});
    test_conversion_to_mdspan(layout_left_padded::mapping<Kokkos::extents<std::size_t, 4, 7>>({}, 4), Kokkos::View<value_type **, Kokkos::LayoutLeft>("v", 4, 7), test_accessor{});

    test_conversion_to_mdspan(layout_right_padded::mapping<Kokkos::extents<std::size_t, 4>>({}, 4), Kokkos::View<value_type *, Kokkos::LayoutRight>("v", 4), test_accessor{});
    test_conversion_to_mdspan(layout_right_padded::mapping<Kokkos::extents<std::size_t, 4, 7>>({}, 7), Kokkos::View<value_type **, Kokkos::LayoutRight>("v", 4, 7), test_accessor{});
  }
};

namespace Test {

TEST(TEST_CATEGORY, view_mdspan_conversion) {
  TestViewMDSpanConversion<double, TEST_EXECSPACE>::run_test();
}

}  // namespace Test

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN
