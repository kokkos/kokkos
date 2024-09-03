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

#include <Kokkos_Core.hpp>
#include <type_traits>
#include "experimental/__p0009_bits/dynamic_extent.hpp"
#include "experimental/__p2642_bits/layout_padded_fwd.hpp"
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#include <View/MDSpan/Kokkos_MDSpan_Accessor.hpp>
#include <View/Kokkos_BasicView.hpp>
#endif  // KOKKOS_IMPL_PUBLIC_INCLUDE

namespace {
template <class IndexType, std::size_t>
IndexType fill_zero() {
  return IndexType(0);
}

template <class ExecutionSpace, class Extents>
auto make_spanning_mdrange_policy_from_extents_impl(const Extents &extents,
                                                    std::index_sequence<0>) {
  using index_type = typename Extents::index_type;
  return Kokkos::RangePolicy<ExecutionSpace>{0, extents.extent(0)};
}

template <class ExecutionSpace, class Extents, std::size_t... Indices>
auto make_spanning_mdrange_policy_from_extents_impl(
    const Extents &extents, std::index_sequence<Indices...>) {
  using index_type    = typename Extents::index_type;
  constexpr auto rank = Extents::rank();
  return Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<rank>>{
      {fill_zero<index_type, Indices>()...}, {extents.extent(Indices)...}};
}

template <class ExecutionSpace, class Extents>
auto make_spanning_mdrange_policy_from_extents(const Extents &extents) {
  return make_spanning_mdrange_policy_from_extents_impl<ExecutionSpace>(
      extents, std::make_index_sequence<Extents::rank()>{});
}

template <class T, class ExecutionSpace>
struct TestBasicView {
  template <class ExtentsType>
  static void test_default_constructor() {
    using extents_type  = ExtentsType;
    using layout_type   = Kokkos::Experimental::layout_right_padded<>;
    using accessor_type = Kokkos::Impl::checked_reference_counted_accessor<
        T, typename ExecutionSpace::memory_space>;
    using view_type =
        Kokkos::BasicView<T, ExtentsType, layout_type, accessor_type>;
    view_type view;

    EXPECT_FALSE(view.data_handle().has_record());
    EXPECT_EQ(view.data_handle().get(), nullptr);
    EXPECT_EQ(view.extents(), extents_type{});
    EXPECT_EQ(view.data_handle().use_count(), 0);
    EXPECT_TRUE(view.is_exhaustive());
    EXPECT_EQ(view.data_handle().get_label(), "");
  }

  template <class ExtentsType>
  static void test_extents_constructor(const ExtentsType &extents) {
    using extents_type  = ExtentsType;
    using layout_type   = Kokkos::Experimental::layout_right_padded<>;
    using accessor_type = Kokkos::Impl::checked_reference_counted_accessor<
        T, typename ExecutionSpace::memory_space>;
    using view_type =
        Kokkos::BasicView<T, ExtentsType, layout_type, accessor_type>;

    view_type view("test_view", extents);

    EXPECT_TRUE(view.data_handle().has_record());
    EXPECT_NE(view.data_handle().get(), nullptr);
    EXPECT_EQ(view.extents(), extents);
    EXPECT_EQ(view.data_handle().use_count(), 1);
    EXPECT_TRUE(view.is_exhaustive());
    EXPECT_EQ(view.data_handle().get_label(), "test_view");
  }

  template <template <std::size_t> class LayoutType, class ExtentsType>
  static void test_mapping_constructor(const ExtentsType &extents,
                                       std::size_t _padding) {
    using extents_type  = ExtentsType;
    using layout_type   = LayoutType<Kokkos::dynamic_extent>;
    using mapping_type  = typename layout_type::template mapping<ExtentsType>;
    using accessor_type = Kokkos::Impl::checked_reference_counted_accessor<
        T, typename ExecutionSpace::memory_space>;
    using view_type =
        Kokkos::BasicView<T, extents_type, layout_type, accessor_type>;
    static_assert(
        std::is_same_v<typename view_type::mapping_type, mapping_type>);

    auto mapping = mapping_type(extents, _padding);

    view_type view("test_view", mapping);

    EXPECT_TRUE(view.data_handle().has_record());
    EXPECT_NE(view.data_handle().get(), nullptr);
    EXPECT_EQ(view.extents(), mapping.extents());
    EXPECT_EQ(view.data_handle().use_count(), 1);
    EXPECT_EQ(view.is_exhaustive(), mapping.is_exhaustive());
    EXPECT_EQ(view.data_handle().get_label(), "test_view");
  }

  template <class ViewType>
  struct MDRangeTestFunctor {
    ViewType view;
    template <class... Idxs>
    KOKKOS_FUNCTION void operator()(Idxs... idxs) const {
      view(idxs...) = (idxs + ...);
    }
  };

  template <class LayoutType, class Extents>
  static void test_access_with_extents(const Extents &extents) {
    using extents_type  = Extents;
    using layout_type   = Kokkos::Experimental::layout_right_padded<>;
    using accessor_type = Kokkos::Impl::checked_reference_counted_accessor<
        T, typename ExecutionSpace::memory_space>;
    using view_type = Kokkos::BasicView<T, Extents, layout_type, accessor_type>;

    auto view = view_type("test_view", extents);

    EXPECT_TRUE(view.data_handle().has_record());
    EXPECT_NE(view.data_handle().get(), nullptr);

    auto mdrange_policy =
        make_spanning_mdrange_policy_from_extents<ExecutionSpace>(extents);

    Kokkos::parallel_for(mdrange_policy, MDRangeTestFunctor{view});
  }

#if 0  // TODO: this test should be active after View is put on top of BasicView
  template <template <std::size_t> class LayoutType, class SrcViewType,
            class ExtentsType>
  static void test_construct_from_view(const ExtentsType &extents,
                                       std::size_t _padding) {
    using extents_type  = ExtentsType;
    using layout_type   = LayoutType<Kokkos::dynamic_extent>;
    using mapping_type  = typename layout_type::template mapping<ExtentsType>;
    using accessor_type = Kokkos::Impl::checked_reference_counted_accessor<
        T, typename ExecutionSpace::memory_space>;
    using basic_view_type =
        Kokkos::BasicView<T, extents_type, layout_type, accessor_type>;
    using view_type = SrcViewType;
    static_assert(std::is_constructible_v<basic_view_type, SrcViewType>);
  }
#endif

  template <class LayoutType>
  static void test_access() {
    test_access_with_extents<LayoutType>(Kokkos::extents<std::size_t, 5>());
    // test_access_with_extents<LayoutType>(Kokkos::extents<std::size_t, 5,
    // 7>()); test_access_with_extents<LayoutType>(Kokkos::extents<std::size_t,
    // Kokkos::dynamic_extent>(12)); test_access_with_extents<LayoutType>(
    //     Kokkos::extents<std::size_t, Kokkos::dynamic_extent, 7>(9));
  }

  static void run_test() {
    test_default_constructor<Kokkos::extents<std::size_t, 1>>();

    test_extents_constructor(
        Kokkos::extents<std::size_t, 2, Kokkos::dynamic_extent, 4>(8));
    test_extents_constructor(Kokkos::extents<std::size_t, 2, 4>());
    test_extents_constructor(Kokkos::extents<std::size_t>());

    test_mapping_constructor<Kokkos::Experimental::layout_left_padded>(
        Kokkos::extents<std::size_t, 2, Kokkos::dynamic_extent>(2, 5), 8);
    test_mapping_constructor<Kokkos::Experimental::layout_left_padded>(
        Kokkos::extents<std::size_t>(), 4);
    test_mapping_constructor<Kokkos::Experimental::layout_left_padded>(
        Kokkos::extents<std::size_t, 2, 3>(), 9);

    test_mapping_constructor<Kokkos::Experimental::layout_right_padded>(
        Kokkos::extents<std::size_t, 2, Kokkos::dynamic_extent>(2, 5), 8);
    test_mapping_constructor<Kokkos::Experimental::layout_right_padded>(
        Kokkos::extents<std::size_t>(), 4);
    test_mapping_constructor<Kokkos::Experimental::layout_right_padded>(
        Kokkos::extents<std::size_t, 2, 3>(), 9);

#if 0  // TODO: this test should be active after View is put on top of BasicView
    test_construct_from_view<
        Kokkos::Experimental::layout_left_padded,
        Kokkos::View<double[3], Kokkos::LayoutLeft, ExecutionSpace>>(
        Kokkos::extents<std::size_t, 3>(), 0);

    test_construct_from_view<
        Kokkos::Experimental::layout_left_padded,
        Kokkos::View<double[3], Kokkos::LayoutLeft, ExecutionSpace>>(
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(3), 0);
#endif

    test_access<
        Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>>();
    test_access<
        Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>>();
  }
};
}  // namespace

namespace Test {

TEST(TEST_CATEGORY, basic_view) {
  TestBasicView<double, TEST_EXECSPACE>::run_test();
}

}  // namespace Test
