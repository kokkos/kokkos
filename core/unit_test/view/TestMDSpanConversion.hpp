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

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN

template <class T, class ExecutionSpace>
struct TestViewMDSpanConversion {
  using value_type = T;

  template <std::size_t Padding>
  using layout_left_padded = Kokkos::Experimental::layout_left_padded<Padding>;

  template <std::size_t Padding>
  using layout_right_padded =
      Kokkos::Experimental::layout_right_padded<Padding>;

  struct test_accessor {
    using offset_policy    = test_accessor;
    using element_type     = value_type;
    using reference        = element_type &;
    using data_handle_type = element_type *;

    constexpr test_accessor() noexcept = default;
    constexpr reference access(data_handle_type p, std::size_t i) {
      return p[i];
    }
    constexpr data_handle_type offset(data_handle_type p, std::size_t i) {
      return p + i;
    }
  };

  template <class KokkosLayout, class DataType, class MDSpanLayoutMapping,
            class... RefViewProps>
  static void test_conversion_from_mdspan(
      Kokkos::View<DataType, RefViewProps...> ref,
      const MDSpanLayoutMapping &mapping) {
    using view_type = Kokkos::View<DataType, KokkosLayout, ExecutionSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using natural_mdspan_type =
        typename Kokkos::Experimental::Impl::MDSpanViewTraits<
            typename view_type::traits>::mdspan_type;
    using mapping_type       = MDSpanLayoutMapping;
    using mdspan_layout_type = typename MDSpanLayoutMapping::layout_type;
    using extents_type       = typename mapping_type::extents_type;
    using mdspan_type =
        Kokkos::mdspan<value_type, extents_type, mdspan_layout_type>;

    static_assert(std::is_convertible_v<mdspan_type, natural_mdspan_type>);

    // Manually create an mdspan from ref so we have a valid pointer to play
    // with
    const auto &exts = mapping.extents();
    auto mds         = mdspan_type{ref.data(), mapping};

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
    auto cvt = v.to_mdspan(a);
    ASSERT_EQ(cvt.data_handle(), v.data());
    ASSERT_EQ(cvt.mapping(), ref_layout_mapping);
  }

  static void run_test() {
    // nvcc doesn't do CTAD properly here, making this way more verbose..
    // LayoutLeft
    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type *, Kokkos::LayoutLeft, ExecutionSpace>("ref",
                                                                       7),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 1>>{
            Kokkos::dextents<std::size_t, 1>(7)});
    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type[7], Kokkos::LayoutLeft, ExecutionSpace>("ref"),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7>>{
            Kokkos::extents<std::size_t, 7>()});
    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type[7], Kokkos::LayoutLeft, ExecutionSpace>("ref"),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 1>>{
            Kokkos::dextents<std::size_t, 1>(7)});
    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type *, Kokkos::LayoutLeft, ExecutionSpace>("ref",
                                                                       7),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7>>{
            Kokkos::extents<std::size_t, 7>()});

    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type **, Kokkos::LayoutLeft, ExecutionSpace>("ref",
                                                                        7, 3),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 2>>{
            Kokkos::dextents<std::size_t, 2>(7, 3)});
    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type[7][3], Kokkos::LayoutLeft, ExecutionSpace>(
            "ref"),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7, 3>>{
            Kokkos::extents<std::size_t, 7, 3>()});
    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type[7][3], Kokkos::LayoutLeft, ExecutionSpace>(
            "ref"),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 2>>{
            Kokkos::dextents<std::size_t, 2>(7, 3)});
    test_conversion_from_mdspan<Kokkos::LayoutLeft>(
        Kokkos::View<value_type **, Kokkos::LayoutLeft, ExecutionSpace>("ref",
                                                                        7, 3),
        typename layout_left_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7, 3>>{
            Kokkos::extents<std::size_t, 7, 3>()});

    // LayoutRight
    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type *, Kokkos::LayoutRight, ExecutionSpace>("ref",
                                                                        7),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 1>>{
            Kokkos::dextents<std::size_t, 1>(7)});
    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type[7], Kokkos::LayoutRight, ExecutionSpace>("ref"),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7>>{
            Kokkos::extents<std::size_t, 7>()});
    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type[7], Kokkos::LayoutRight, ExecutionSpace>("ref"),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 1>>{
            Kokkos::dextents<std::size_t, 1>(7)});
    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type *, Kokkos::LayoutRight, ExecutionSpace>("ref",
                                                                        7),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7>>{
            Kokkos::extents<std::size_t, 7>()});

    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type **, Kokkos::LayoutRight, ExecutionSpace>("ref",
                                                                         7, 3),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 2>>{
            Kokkos::dextents<std::size_t, 2>(7, 3)});
    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type[7][3], Kokkos::LayoutRight, ExecutionSpace>(
            "ref"),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7, 3>>{
            Kokkos::extents<std::size_t, 7, 3>()});
    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type[7][3], Kokkos::LayoutRight, ExecutionSpace>(
            "ref"),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::dextents<std::size_t, 2>>{
            Kokkos::dextents<std::size_t, 2>(7, 3)});
    test_conversion_from_mdspan<Kokkos::LayoutRight>(
        Kokkos::View<value_type **, Kokkos::LayoutRight, ExecutionSpace>("ref",
                                                                         7, 3),
        typename layout_right_padded<sizeof(
            value_type)>::template mapping<Kokkos::extents<std::size_t, 7, 3>>{
            Kokkos::extents<std::size_t, 7, 3>()});

    // LayoutStride
    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type *, Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2}),
        Kokkos::layout_stride::mapping<Kokkos::dextents<std::size_t, 1>>{
            Kokkos::dextents<std::size_t, 1>{7},
            std::array<std::size_t, 1>{2}});
    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type[7], Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2}),
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7>>{
            {}, std::array<std::size_t, 1>{2}});
    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type[7], Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2}),
        Kokkos::layout_stride::mapping<Kokkos::dextents<std::size_t, 1>>{
            Kokkos::dextents<std::size_t, 1>{7},
            std::array<std::size_t, 1>{2}});
    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type *, Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2}),
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7>>{
            Kokkos::extents<std::size_t, 7>(), std::array<std::size_t, 1>{2}});

    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type **, Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2, 3, 4}),
        Kokkos::layout_stride::mapping<Kokkos::dextents<std::size_t, 2>>{
            Kokkos::dextents<std::size_t, 2>(7, 3),
            std::array<std::size_t, 2>{2, 4}});
    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type[7][3], Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2, 3, 4}),
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 3>>{
            Kokkos::extents<std::size_t, 7, 3>(),
            std::array<std::size_t, 2>{2, 4}});
    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type[7][3], Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2, 3, 4}),
        Kokkos::layout_stride::mapping<Kokkos::dextents<std::size_t, 2>>{
            Kokkos::dextents<std::size_t, 2>(7, 3),
            std::array<std::size_t, 2>{2, 4}});
    test_conversion_from_mdspan<Kokkos::LayoutStride>(
        Kokkos::View<value_type **, Kokkos::LayoutStride, ExecutionSpace>(
            "ref", Kokkos::LayoutStride{7, 2, 3, 4}),
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 3>>{
            Kokkos::extents<std::size_t, 7, 3>(),
            std::array<std::size_t, 2>{2, 4}});

    // Conversion to mdspan
    test_conversion_to_mdspan(
        layout_left_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4>>({}, 4),
        Kokkos::View<value_type *, Kokkos::LayoutLeft, ExecutionSpace>("v", 4));
    test_conversion_to_mdspan(
        layout_left_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4, 7>>({}, 4),
        Kokkos::View<value_type **, Kokkos::LayoutLeft, ExecutionSpace>("v", 4,
                                                                        7));

    test_conversion_to_mdspan(
        layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4>>({}, 4),
        Kokkos::View<value_type *, Kokkos::LayoutRight, ExecutionSpace>("v",
                                                                        4));
    test_conversion_to_mdspan(
        layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4, 7>>({}, 7),
        Kokkos::View<value_type **, Kokkos::LayoutRight, ExecutionSpace>("v", 4,
                                                                         7));

    test_conversion_to_mdspan(
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 4>>(
            {}, std::array<std::size_t, 1>{5}),
        Kokkos::View<value_type *, Kokkos::LayoutStride, ExecutionSpace>(
            "v", Kokkos::LayoutStride{4, 5}));
    test_conversion_to_mdspan(
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 4, 7>>(
            {}, std::array<std::size_t, 2>{5, 9}),
        Kokkos::View<value_type **, Kokkos::LayoutStride, ExecutionSpace>(
            "v", Kokkos::LayoutStride{4, 5, 7, 9}));

    test_conversion_to_mdspan(
        layout_left_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4>>({}, 4),
        Kokkos::View<value_type *, Kokkos::LayoutLeft, ExecutionSpace>("v", 4),
        Kokkos::default_accessor<value_type>{});
    test_conversion_to_mdspan(
        layout_left_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4, 7>>({}, 4),
        Kokkos::View<value_type **, Kokkos::LayoutLeft, ExecutionSpace>("v", 4,
                                                                        7),
        Kokkos::default_accessor<value_type>{});

    test_conversion_to_mdspan(
        layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4>>({}, 4),
        Kokkos::View<value_type *, Kokkos::LayoutRight, ExecutionSpace>("v", 4),
        Kokkos::default_accessor<value_type>{});
    test_conversion_to_mdspan(
        layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4, 7>>({}, 7),
        Kokkos::View<value_type **, Kokkos::LayoutRight, ExecutionSpace>("v", 4,
                                                                         7),
        Kokkos::default_accessor<value_type>{});

    test_conversion_to_mdspan(
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 4>>(
            {}, std::array<std::size_t, 1>{5}),
        Kokkos::View<value_type *, Kokkos::LayoutStride, ExecutionSpace>(
            "v", Kokkos::LayoutStride{4, 5}));
    test_conversion_to_mdspan(
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 4, 7>>(
            {}, std::array<std::size_t, 2>{5, 9}),
        Kokkos::View<value_type **, Kokkos::LayoutStride, ExecutionSpace>(
            "v", Kokkos::LayoutStride{4, 5, 7, 9}),
        Kokkos::default_accessor<value_type>{});

    test_conversion_to_mdspan(
        layout_left_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4>>({}, 4),
        Kokkos::View<value_type *, Kokkos::LayoutLeft, ExecutionSpace>("v", 4),
        test_accessor{});
    test_conversion_to_mdspan(
        layout_left_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4, 7>>({}, 4),
        Kokkos::View<value_type **, Kokkos::LayoutLeft, ExecutionSpace>("v", 4,
                                                                        7),
        test_accessor{});

    test_conversion_to_mdspan(
        layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4>>({}, 4),
        Kokkos::View<value_type *, Kokkos::LayoutRight, ExecutionSpace>("v", 4),
        test_accessor{});
    test_conversion_to_mdspan(
        layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::size_t, 4, 7>>({}, 7),
        Kokkos::View<value_type **, Kokkos::LayoutRight, ExecutionSpace>("v", 4,
                                                                         7),
        test_accessor{});

    test_conversion_to_mdspan(
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 4>>(
            {}, std::array<std::size_t, 1>{5}),
        Kokkos::View<value_type *, Kokkos::LayoutStride, ExecutionSpace>(
            "v", Kokkos::LayoutStride{4, 5}));
    test_conversion_to_mdspan(
        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 4, 7>>(
            {}, std::array<std::size_t, 2>{5, 9}),
        Kokkos::View<value_type **, Kokkos::LayoutStride, ExecutionSpace>(
            "v", Kokkos::LayoutStride{4, 5, 7, 9}),
        test_accessor{});
  }
};

namespace Test {

TEST(TEST_CATEGORY, view_mdspan_conversion) {
  TestViewMDSpanConversion<double, TEST_EXECSPACE>::run_test();
  TestViewMDSpanConversion<float, TEST_EXECSPACE>::run_test();
  TestViewMDSpanConversion<int, TEST_EXECSPACE>::run_test();
}

}  // namespace Test

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN
