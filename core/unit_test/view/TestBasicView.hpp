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
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#include <View/Kokkos_BasicView.hpp>

template <class T, class ExecutionSpace>
struct TestBasicView
{
  template <class ExtentsType>
  static void test_default_constructor() {
    using extents_type = ExtentsType;
    using view_type    = Kokkos::BasicView<T, ExtentsType, ExecutionSpace>;
    view_type view;

    EXPECT_EQ(view.is_allocated(), false);
    EXPECT_EQ(view.data(), nullptr);
    EXPECT_EQ(view.extents(), extents_type{});
    EXPECT_EQ(view.use_count(), 0);
    EXPECT_EQ(view.span_is_contiguous(), true);
    EXPECT_EQ(view.label(), "");
  }

  template <class ExtentsType>
  static void test_extents_constructor(const ExtentsType &extents) {
    using extents_type = ExtentsType;
    using view_type    = Kokkos::BasicView<T, ExtentsType, ExecutionSpace>;

    view_type view("test_view", extents);

    EXPECT_EQ(view.is_allocated(), true);
    EXPECT_NE(view.data(), nullptr);
    EXPECT_EQ(view.extents(), extents);
    EXPECT_EQ(view.use_count(), 1);
    EXPECT_EQ(view.span_is_contiguous(), true);
    EXPECT_EQ(view.label(), "test_view");
  }

  template <template < std::size_t > class LayoutType, class ExtentsType>
  static void test_mapping_constructor(const ExtentsType &extents, std::size_t _padding) {
    using extents_type = ExtentsType;
    using layout_type  = LayoutType<Kokkos::dynamic_extent>;
    using mapping_type = typename layout_type::mapping<ExtentsType>;
    using view_type = Kokkos::BasicView<T, extents_type, ExecutionSpace, typename ExecutionSpace::memory_space, layout_type>;
    static_assert(std::is_same_v<typename view_type::mapping_type, mapping_type>);

    auto mapping = mapping_type(extents, _padding);

    view_type view("test_view", mapping);

    EXPECT_EQ(view.is_allocated(), true);
    EXPECT_NE(view.data(), nullptr);
    EXPECT_EQ(view.extents(), mapping.extents());
    EXPECT_EQ(view.use_count(), 1);
    EXPECT_EQ(view.span_is_contiguous(), mapping.is_exhaustive());
    EXPECT_EQ(view.label(), "test_view");
  }

  template <class ExtentsType, class... IndexTypes>
  static void test_indices_constructor(ExtentsType cmp_extents, IndexTypes... indices) {
    using extents_type = ExtentsType;
    using view_type    = Kokkos::BasicView<T, ExtentsType, ExecutionSpace>;

    view_type view("test_view", indices...);

    EXPECT_EQ(view.is_allocated(), true);
    EXPECT_NE(view.data(), nullptr);
    EXPECT_EQ(view.extents(), cmp_extents);
    EXPECT_EQ(view.use_count(), 1);
    EXPECT_EQ(view.span_is_contiguous(), true);
    EXPECT_EQ(view.label(), "test_view");
  }

  template <class LayoutType, class Extents>
  static void test_access_with_extents(const Extents &extents) {
    using extents_type = Extents;
    using view_type = Kokkos::BasicView<T, extents_type, ExecutionSpace>;

    auto view = view_type("test_view", extents);

    auto mdrange_policy =
        Kokkos::Experimental::Impl::make_spanning_mdrange_policy_from_extents<ExecutionSpace>(extents);

    Kokkos::parallel_for(mdrange_policy, KOKKOS_LAMBDA(auto... idxs) {
      view(idxs...) = (idxs + ...);
    } );
  }

  template <class LayoutType>
  static void test_access()
  {
    test_access_with_extents<LayoutType>(Kokkos::extents<std::size_t, 5>());
    test_access_with_extents<LayoutType>(Kokkos::extents<std::size_t, 5, 7>());
    test_access_with_extents<LayoutType>(Kokkos::extents<std::size_t, Kokkos::dynamic_extent>(12));
    test_access_with_extents<LayoutType>(
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent, 7>(9));
  }

  static void run_test() {
    test_default_constructor<Kokkos::extents<std::size_t, 1>>();

    test_extents_constructor(
        Kokkos::extents<std::size_t, 2, Kokkos::dynamic_extent, 4>(8));
    test_extents_constructor(Kokkos::extents<std::size_t, 2, 4>());
    test_extents_constructor(Kokkos::extents<std::size_t>());

    test_indices_constructor(
        Kokkos::extents<std::size_t, 2, Kokkos::dynamic_extent, 4>{2, 3, 4}, 3);
    test_indices_constructor(
        Kokkos::extents<std::size_t, 2, Kokkos::dynamic_extent, 4>{2, 3, 4}, 2,
        3, 4);
    test_indices_constructor(Kokkos::extents<std::size_t, 2, 4>{2, 4});
    test_indices_constructor(Kokkos::extents<std::size_t, 2, 4>{2, 4}, 2, 4);
    test_indices_constructor(Kokkos::extents<std::size_t>{});
    test_indices_constructor(
        Kokkos::extents<std::size_t, Kokkos::dynamic_extent,
                        Kokkos::dynamic_extent, Kokkos::dynamic_extent>{5, 6,
                                                                        7},
        5, 6.3, 7ULL);

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

    test_access<Kokkos::Experimental::layout_left_padded<Kokkos::dynamic_extent>>();
    test_access<Kokkos::Experimental::layout_right_padded<Kokkos::dynamic_extent>>();
  }
};

namespace Test {

TEST(TEST_CATEGORY, basic_view) {
  TestBasicView<double, TEST_EXECSPACE>::run_test();
}

}  // namespace Test

#endif // KOKKOS_IMPL_PUBLIC_INCLUDE
