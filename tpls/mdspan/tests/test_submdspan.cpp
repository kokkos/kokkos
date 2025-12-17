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
#include <mdspan/mdspan.hpp>
#include <vector>

#include <gtest/gtest.h>

#include <utility>
#include <numeric>

#include "offload_utils.hpp"
#include "foo_customizations.hpp"

MDSPAN_IMPL_INLINE_VARIABLE constexpr auto dyn = Kokkos::dynamic_extent;

TEST(TestSubmdspanLayoutRightStaticSizedRankReducing3Dto1D, test_submdspan_layout_right_static_sized_rank_reducing_3d_to_1d) {
  std::vector<int> d(2 * 3 * 4, 0);
  Kokkos::mdspan<int, Kokkos::extents<size_t,2, 3, 4>> m(d.data());
  MDSPAN_IMPL_OP(m, 1, 1, 1) = 42;
  auto sub0 = Kokkos::submdspan(m, 1, 1, Kokkos::full_extent);
  static_assert(decltype(sub0)::rank()==1,"unexpected submdspan rank");
  ASSERT_EQ(sub0.rank(),         1);
  ASSERT_EQ(sub0.rank_dynamic(), 0);
  ASSERT_EQ(sub0.extent(0),      4);
  ASSERT_EQ((MDSPAN_IMPL_OP(sub0, 1)), 42);
}

TEST(TestSubmdspanLayoutLeftStaticSizedRankReducing3Dto1D, test_submdspan_layout_left_static_sized_rank_reducing_3d_to_1d) {
  std::vector<int> d(2 * 3 * 4, 0);
  Kokkos::mdspan<int, Kokkos::extents<size_t,2, 3, 4>, Kokkos::layout_left> m(d.data());
  MDSPAN_IMPL_OP(m, 1, 1, 1) = 42;
  auto sub0 = Kokkos::submdspan(m, 1, 1, Kokkos::full_extent);
  ASSERT_EQ(sub0.rank(),         1);
  ASSERT_EQ(sub0.rank_dynamic(), 0);
  ASSERT_EQ(sub0.extent(0),      4);
  ASSERT_EQ((MDSPAN_IMPL_OP(sub0, 1)), 42);
}

TEST(TestSubmdspanLayoutRightStaticSizedRankReducingNested3Dto0D, test_submdspan_layout_right_static_sized_rank_reducing_nested_3d_to_0d) {
  std::vector<int> d(2 * 3 * 4, 0);
  Kokkos::mdspan<int, Kokkos::extents<size_t,2, 3, 4>> m(d.data());
  MDSPAN_IMPL_OP(m, 1, 1, 1) = 42;
  auto sub0 = Kokkos::submdspan(m, 1, Kokkos::full_extent, Kokkos::full_extent);
  ASSERT_EQ(sub0.rank(),         2);
  ASSERT_EQ(sub0.rank_dynamic(), 0);
  ASSERT_EQ(sub0.extent(0),      3);
  ASSERT_EQ(sub0.extent(1),      4);
  ASSERT_EQ((MDSPAN_IMPL_OP(sub0, 1, 1)), 42);
  auto sub1 = Kokkos::submdspan(sub0, 1, Kokkos::full_extent);
  ASSERT_EQ(sub1.rank(),         1);
  ASSERT_EQ(sub1.rank_dynamic(), 0);
  ASSERT_EQ(sub1.extent(0),      4);
  ASSERT_EQ((MDSPAN_IMPL_OP(sub1,1)),42);
  auto sub2 = Kokkos::submdspan(sub1, 1);
  ASSERT_EQ(sub2.rank(),         0);
  ASSERT_EQ(sub2.rank_dynamic(), 0);
  ASSERT_EQ((MDSPAN_IMPL_OP0(sub2)), 42);
}

TEST(TestSubmdspanLayoutRightStaticSizedPairs, test_submdspan_layout_right_static_sized_pairs) {
  std::vector<int> d(2 * 3 * 4, 0);
  Kokkos::mdspan<int, Kokkos::extents<size_t,2, 3, 4>> m(d.data());
  MDSPAN_IMPL_OP(m, 1, 1, 1) = 42;
  auto sub0 = Kokkos::submdspan(m, std::pair<int,int>{1, 2}, std::pair<int,int>{1, 3}, std::pair<int,int>{1, 4});
  ASSERT_EQ(sub0.rank(),         3);
  ASSERT_EQ(sub0.rank_dynamic(), 3);
  ASSERT_EQ(sub0.extent(0),      1);
  ASSERT_EQ(sub0.extent(1),      2);
  ASSERT_EQ(sub0.extent(2),      3);
  ASSERT_EQ((MDSPAN_IMPL_OP(sub0, 0, 0, 0)), 42);
}

TEST(TestSubmdspanLayoutRightStaticSizedTuples, test_submdspan_layout_right_static_sized_tuples) {
  std::vector<int> d(2 * 3 * 4, 0);
  Kokkos::mdspan<int, Kokkos::extents<size_t,2, 3, 4>> m(d.data());
  MDSPAN_IMPL_OP(m, 1, 1, 1) = 42;
  auto sub0 = Kokkos::submdspan(m, std::tuple<int,int>{1, 2}, std::tuple<int,int>{1, 3}, std::tuple<int,int>{1, 4});
  ASSERT_EQ(sub0.rank(),         3);
  ASSERT_EQ(sub0.rank_dynamic(), 3);
  ASSERT_EQ(sub0.extent(0),      1);
  ASSERT_EQ(sub0.extent(1),      2);
  ASSERT_EQ(sub0.extent(2),      3);
  ASSERT_EQ((MDSPAN_IMPL_OP(sub0, 0, 0, 0)),       42);
}


//template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>


template<size_t ... Args>
using args_t = std::index_sequence<Args...>;

template<size_t PaddingValue>
using layout_left_padded = Kokkos::Experimental::layout_left_padded<PaddingValue>;
template<size_t PaddingValue>
using layout_right_padded = Kokkos::Experimental::layout_right_padded<PaddingValue>;

using submdspan_test_types =
  ::testing::Types<
      // layout_left to layout_left
      std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,1>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,1>, std::pair<int,int>>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,0>, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, std::pair<int,int>>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,1>, Kokkos::full_extent_t, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,3>, args_t<10,20,30>,    Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, Kokkos::full_extent_t, std::pair<int,int>>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,3>, args_t<10,20,30>,    Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, std::pair<int,int>, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,3>, args_t<10,20,30>,    Kokkos::dextents<size_t,1>, Kokkos::full_extent_t, int, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,3>, args_t<10,20,30>,    Kokkos::dextents<size_t,1>, std::pair<int,int>, int, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,6>, args_t<6,4,5,6,7,8>, Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, Kokkos::full_extent_t, std::pair<int,int>, int, int, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,6>, args_t<6,4,5,6,7,8>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, std::pair<int,int>, int, int, int, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,6>, args_t<6,4,5,6,7,8>, Kokkos::dextents<size_t,1>, Kokkos::full_extent_t, int, int, int ,int, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_left, Kokkos::dextents<size_t,6>, args_t<6,4,5,6,7,8>, Kokkos::dextents<size_t,1>, std::pair<int,int>, int, int, int, int, int>
    // layout_right to layout_right
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,1>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,1>, std::pair<int,int>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,0>, int>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,2>, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,1>, int, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,3>, args_t<10,20,30>,    Kokkos::dextents<size_t,3>, std::pair<int,int>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,3>, args_t<10,20,30>,    Kokkos::dextents<size_t,2>, int, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,3>, args_t<10,20,30>,    Kokkos::dextents<size_t,1>, int, int, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,6>, args_t<6,4,5,6,7,8>, Kokkos::dextents<size_t,3>, int, int, int, std::pair<int,int>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,6>, args_t<6,4,5,6,7,8>, Kokkos::dextents<size_t,2>, int, int, int, int, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::dextents<size_t,6>, args_t<6,4,5,6,7,8>, Kokkos::dextents<size_t,1>, int, int, int, int, int, Kokkos::full_extent_t>
    // layout_right to layout_right Check Extents Preservation
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10>,           args_t<10>,          Kokkos::extents<size_t,10>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10>,           args_t<10>,          Kokkos::extents<size_t,dyn>, std::pair<int,int>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10>,           args_t<10>,          Kokkos::extents<size_t,dyn>, std::complex<double>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10>,           args_t<10>,          Kokkos::extents<size_t>, int>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10,20>,        args_t<10,20>,       Kokkos::extents<size_t,10,20>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10,20>,        args_t<10,20>,       Kokkos::extents<size_t,dyn,20>, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10,20>,        args_t<10,20>,       Kokkos::extents<size_t,20>, int, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10,20,30>,     args_t<10,20,30>,    Kokkos::extents<size_t,dyn,20,30>, std::pair<int,int>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10,20,30>,     args_t<10,20,30>,    Kokkos::extents<size_t,dyn,30>, int, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,10,20,30>,     args_t<10,20,30>,    Kokkos::extents<size_t,30>, int, int, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,6,4,5,6,7,8>,  args_t<6,4,5,6,7,8>, Kokkos::extents<size_t,dyn,7,8>, int, int, int, std::pair<int,int>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,6,4,5,6,7,8>,  args_t<6,4,5,6,7,8>, Kokkos::extents<size_t,dyn,8>, int, int, int, int, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_right, Kokkos::extents<size_t,6,4,5,6,7,8>,  args_t<6,4,5,6,7,8>, Kokkos::extents<size_t,8>, int, int, int, int, int, Kokkos::full_extent_t>
    // layout_left to layout_left_padded
    , std::tuple<Kokkos::layout_left, layout_left_padded<dyn>, Kokkos::dextents<size_t,2>, args_t<10,20>,          Kokkos::dextents<size_t,2>, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_left, layout_left_padded<10>,  Kokkos::extents<size_t,10,dyn>, args_t<10,20>,      Kokkos::dextents<size_t,2>, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_left, layout_left_padded<dyn>, Kokkos::dextents<size_t,3>, args_t<10,20,30>,       Kokkos::dextents<size_t,2>, std::pair<int,int>, int, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_left, layout_left_padded<dyn>, Kokkos::dextents<size_t,4>, args_t<10,20,30,40>,    Kokkos::dextents<size_t,3>, std::pair<int,int>, int, Kokkos::full_extent_t, std::pair<int,int>>
    , std::tuple<Kokkos::layout_left, layout_left_padded<dyn>, Kokkos::dextents<size_t,5>, args_t<10,20,30,40,50>, Kokkos::dextents<size_t,3>, std::pair<int,int>, int, Kokkos::full_extent_t, std::pair<int,int>, int>
    // layout_left to layout_stride
    , std::tuple<Kokkos::layout_left, Kokkos::layout_stride,  Kokkos::dextents<size_t,1>,          args_t<10>,          Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_stride,  Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>, int>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_stride,  Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,2>, std::pair<int,int>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_stride,  Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::strided_slice<int,int,int>, std::pair<int,int>>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_stride,  Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::strided_slice<int,int,int>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_stride,  Kokkos::extents<size_t,6,4,5,6,7,8>, args_t<6,4,5,6,7,8>, Kokkos::extents<size_t,6,dyn,8>, Kokkos::full_extent_t, int, std::pair<int,int>, int, int, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_left, Kokkos::layout_stride,  Kokkos::extents<size_t,6,4,5,6,7,8>, args_t<6,4,5,6,7,8>, Kokkos::extents<size_t,4,dyn,7>, int, Kokkos::full_extent_t, std::pair<int,int>, int, Kokkos::full_extent_t, int>
    // layout_right to layout_right_padded
    , std::tuple<Kokkos::layout_right, layout_right_padded<dyn>, Kokkos::dextents<size_t,2>,       args_t<10,20>,          Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, std::pair<int,int>>
    , std::tuple<Kokkos::layout_right, layout_right_padded<20>,  Kokkos::extents<size_t,dyn,20>,   args_t<10,20>,          Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, std::pair<int,int>>
    , std::tuple<Kokkos::layout_right, layout_right_padded<dyn>, Kokkos::dextents<size_t,3>,       args_t<10,20,30>,       Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, int, std::pair<int,int>>
    , std::tuple<Kokkos::layout_right, layout_right_padded<dyn>, Kokkos::dextents<size_t,4>,       args_t<10,20,30,40>,    Kokkos::dextents<size_t,3>, std::pair<int,int>, Kokkos::full_extent_t, int, std::pair<int,int>>
    , std::tuple<Kokkos::layout_right, layout_right_padded<dyn>, Kokkos::dextents<size_t,5>,       args_t<10,20,30,40,50>, Kokkos::dextents<size_t,3>, int, std::pair<int,int>, Kokkos::full_extent_t, int, std::pair<int,int>>
    // layout_right to layout_stride
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::dextents<size_t,1>,          args_t<10>,          Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::dextents<size_t,1>,          args_t<10>,          Kokkos::extents<size_t,0>,  Kokkos::strided_slice<int,std::integral_constant<int,0>,std::integral_constant<int,0>>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>, int>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,2>, std::pair<int,int>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::strided_slice<int,int,int>, std::pair<int,int>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::strided_slice<int,int,int>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::extents<size_t,6,4,5,6,7,8>, args_t<6,4,5,6,7,8>, Kokkos::extents<size_t,6,dyn,8>, Kokkos::full_extent_t, int, std::pair<int,int>, int, int, Kokkos::full_extent_t>
    , std::tuple<Kokkos::layout_right, Kokkos::layout_stride, Kokkos::extents<size_t,6,4,5,6,7,8>, args_t<6,4,5,6,7,8>, Kokkos::extents<size_t,4,dyn,7>, int, Kokkos::full_extent_t, std::pair<int,int>, int, Kokkos::full_extent_t, int>
    // layout_left_padded to layout_left
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_left, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,0>, int, int>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_left, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,1>, std::pair<int,int>, int>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_left, Kokkos::extents<size_t,10,dyn>, args_t<10,20>, Kokkos::extents<size_t,10>, Kokkos::full_extent_t, int>
    , std::tuple<layout_left_padded<4>, Kokkos::layout_left, Kokkos::dextents<size_t,3>, args_t<10,20,30>, Kokkos::dextents<size_t,1>, std::pair<int,int>, int, int>
    , std::tuple<layout_left_padded<4>, Kokkos::layout_left, Kokkos::extents<size_t,10,dyn,dyn>, args_t<10,20,30>, Kokkos::extents<size_t,10>, Kokkos::full_extent_t, int, int>
    // layout_left_padded to layout_left_padded
    , std::tuple<layout_left_padded<dyn>, layout_left_padded<dyn>, Kokkos::dextents<size_t,0>,        args_t<>, Kokkos::dextents<size_t,0>>
    , std::tuple<layout_left_padded<4>,   layout_left_padded<4>,   Kokkos::dextents<size_t,0>,        args_t<>, Kokkos::dextents<size_t,0>>
    , std::tuple<layout_left_padded<dyn>, layout_left_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<4>,   layout_left_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, std::pair<int, int>, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<4>,   layout_left_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, std::pair<int, int>>
    , std::tuple<layout_left_padded<dyn>, layout_left_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, std::pair<int, int>, std::pair<int, int>>
    , std::tuple<layout_left_padded<12>,  layout_left_padded<12>,  Kokkos::extents<size_t,10,20>,     args_t<10,20>, Kokkos::extents<size_t, dyn, 20>, std::pair<int, int>, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<dyn>, layout_left_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, int, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<4>,   layout_left_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, std::pair<int, int>, int, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<4>,   layout_left_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, int, std::pair<int, int>>
    , std::tuple<layout_left_padded<dyn>, layout_left_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, std::pair<int, int>, int, std::pair<int, int>>
    , std::tuple<layout_left_padded<12>,  layout_left_padded<240>, Kokkos::extents<size_t,10,20,dyn>, args_t<10,20,30>, Kokkos::extents<size_t, dyn, dyn>, std::pair<int, int>, int, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<dyn>, layout_left_padded<dyn>, Kokkos::dextents<size_t,4>,        args_t<10,20,30,40>, Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, int, Kokkos::full_extent_t, Kokkos::full_extent_t>
    // layout_left_padded to layout_stride
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,1>, args_t<10>, Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>, int>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::strided_slice<int,int,int>, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, Kokkos::strided_slice<int,int,int>>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,3>, args_t<10,20,30>, Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, Kokkos::strided_slice<int,int,int>, Kokkos::full_extent_t>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,3>, args_t<10,20,30>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, int, Kokkos::strided_slice<int,int,int>>
    , std::tuple<layout_left_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,4>, args_t<10,20,30,40>, Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, Kokkos::full_extent_t, int, Kokkos::full_extent_t>
    // layout_right_padded to layout_right
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_right, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,0>, int, int>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_right, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,1>, int, std::pair<int,int>>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_right, Kokkos::extents<size_t,dyn,30>, args_t<10,20>, Kokkos::extents<size_t,30>, int, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<4>, Kokkos::layout_right, Kokkos::dextents<size_t,3>, args_t<10,20,30>, Kokkos::dextents<size_t,1>, int, int, std::pair<int,int>>
    , std::tuple<layout_right_padded<4>, Kokkos::layout_right, Kokkos::extents<size_t,dyn,dyn,30>, args_t<10,20,30>, Kokkos::extents<size_t,30>, int, int, Kokkos::full_extent_t>
    // layout_right_padded to layout_right_padded
    , std::tuple<layout_right_padded<dyn>, layout_right_padded<dyn>, Kokkos::dextents<size_t,0>,        args_t<>, Kokkos::dextents<size_t,0>>
    , std::tuple<layout_right_padded<4>,   layout_right_padded<4>,   Kokkos::dextents<size_t,0>,        args_t<>, Kokkos::dextents<size_t,0>>
    , std::tuple<layout_right_padded<dyn>, layout_right_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<4>,   layout_right_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, std::pair<int, int>, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<4>,   layout_right_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, std::pair<int, int>>
    , std::tuple<layout_right_padded<dyn>, layout_right_padded<dyn>, Kokkos::dextents<size_t,2>,        args_t<10,20>, Kokkos::dextents<size_t,2>, std::pair<int, int>, std::pair<int, int>>
    , std::tuple<layout_right_padded<22>,  layout_right_padded<22>,  Kokkos::extents<size_t,10,20>,     args_t<10,20>, Kokkos::extents<size_t, 10, dyn>, Kokkos::full_extent_t, std::pair<int, int>>
    , std::tuple<layout_right_padded<dyn>, layout_right_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, int, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<4>,   layout_right_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, std::pair<int, int>, int, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<4>,   layout_right_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, int, std::pair<int, int>>
    , std::tuple<layout_right_padded<dyn>, layout_right_padded<dyn>, Kokkos::dextents<size_t,3>,        args_t<10,20,30>, Kokkos::dextents<size_t,2>, std::pair<int, int>, int, std::pair<int, int>>
    , std::tuple<layout_right_padded<32>,  layout_right_padded<640>, Kokkos::extents<size_t,dyn,20,30>, args_t<10,20,30>, Kokkos::extents<size_t, dyn, dyn>, Kokkos::full_extent_t, int, std::pair<int, int>>
    , std::tuple<layout_right_padded<dyn>, layout_right_padded<dyn>, Kokkos::dextents<size_t,4>,        args_t<10,20,30,40>, Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, Kokkos::full_extent_t, int, Kokkos::full_extent_t>
    // layout_right_padded to layout_stride
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,1>, args_t<10>, Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,1>, int, Kokkos::strided_slice<int,int,int>>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::strided_slice<int,int,int>, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,2>, args_t<10,20>, Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, Kokkos::strided_slice<int,int,int>>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,3>, args_t<10,20,30>, Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, Kokkos::strided_slice<int,int,int>, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,3>, args_t<10,20,30>, Kokkos::dextents<size_t,2>, Kokkos::strided_slice<int,int,int>, int, Kokkos::full_extent_t>
    , std::tuple<layout_right_padded<dyn>, Kokkos::layout_stride, Kokkos::dextents<size_t,4>, args_t<10,20,30,40>, Kokkos::dextents<size_t,3>, Kokkos::full_extent_t, int, Kokkos::full_extent_t, Kokkos::full_extent_t>
    // Testing of customization point design
    , std::tuple<Foo::layout_foo, Foo::layout_foo, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,1>, Kokkos::full_extent_t>
    , std::tuple<Foo::layout_foo, Foo::layout_foo, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,1>, std::pair<int,int>>
    , std::tuple<Foo::layout_foo, Foo::layout_foo, Kokkos::dextents<size_t,1>, args_t<10>,          Kokkos::dextents<size_t,0>, int>
    , std::tuple<Foo::layout_foo, Foo::layout_foo, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,2>, Kokkos::full_extent_t, Kokkos::full_extent_t>
    , std::tuple<Foo::layout_foo, Foo::layout_foo, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,2>, std::pair<int,int>, Kokkos::full_extent_t>
    , std::tuple<Foo::layout_foo, Foo::layout_foo, Kokkos::dextents<size_t,2>, args_t<10,20>,       Kokkos::dextents<size_t,1>, int, Kokkos::full_extent_t>
    , std::tuple<Foo::layout_foo, Kokkos::layout_stride, Kokkos::dextents<size_t,1>,          args_t<10>,          Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>>
    , std::tuple<Foo::layout_foo, Kokkos::layout_stride, Kokkos::dextents<size_t,2>,          args_t<10,20>,       Kokkos::dextents<size_t,1>, Kokkos::strided_slice<int,int,int>, int>
    >;

template<class T> struct TestSubMDSpan;

template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs, size_t ... ConstrArgs>
struct TestSubMDSpan<
  std::tuple<LayoutOrg,
             LayoutSub,
             ExtentsOrg,
	     std::index_sequence<ConstrArgs...>,
             ExtentsSub,
             SubArgs...>>
  : public ::testing::Test {

  using mds_org_t = Kokkos::mdspan<int, ExtentsOrg, LayoutOrg>;
  using mds_sub_t = Kokkos::mdspan<int, ExtentsSub, LayoutSub>;
  using map_t = typename mds_org_t::mapping_type;

  using mds_sub_deduced_t = decltype(Kokkos::submdspan(mds_org_t(nullptr, map_t()), SubArgs()...));
  using sub_args_t = std::tuple<SubArgs...>;

  MDSPAN_INLINE_FUNCTION
  static int create_slice_arg(int) {
    return 2;
  }
  MDSPAN_INLINE_FUNCTION
  static auto create_slice_arg(std::pair<int,int>) {
    return std::pair<int,int>(1,3);
  }
  MDSPAN_INLINE_FUNCTION
  static auto create_slice_arg(std::complex<double>) {
    return std::complex<double>{1.,3.};
  }
  MDSPAN_INLINE_FUNCTION
  static auto create_slice_arg(Kokkos::strided_slice<int,int,int>) {
    return Kokkos::strided_slice<int,int,int>{1,3,2};
  }
  template<int Ext, int Stride>
  MDSPAN_INLINE_FUNCTION
  static auto create_slice_arg(Kokkos::strided_slice<int,std::integral_constant<int, Ext>, std::integral_constant<int, Stride>>) {
    return Kokkos::strided_slice<int,std::integral_constant<int, Ext>, std::integral_constant<int, Stride>>{1,std::integral_constant<int, Ext>(), std::integral_constant<int, Ext>()};
  }
  MDSPAN_INLINE_FUNCTION
  static auto create_slice_arg(Kokkos::full_extent_t) {
    return Kokkos::full_extent;
  }

  template<class SrcMDSpan, class SubMDSpan, size_t ... SrcIdx, size_t ... SubIdx, class ... SliceArgs>
  MDSPAN_INLINE_FUNCTION
  static bool check_submdspan_match(int src_idx, int sub_idx, SrcMDSpan src_mds, SubMDSpan sub_mds, std::index_sequence<SrcIdx...>, std::index_sequence<SubIdx...>, int, SliceArgs ... slices) {
    return check_submdspan_match(++src_idx, sub_idx, src_mds, sub_mds, std::index_sequence<SrcIdx...,2>(), std::index_sequence<SubIdx...>(), slices...);
  }
  template<class SrcMDSpan, class SubMDSpan, size_t ... SrcIdx, size_t ... SubIdx, class ... SliceArgs>
  MDSPAN_INLINE_FUNCTION
  static bool check_submdspan_match(int src_idx, int sub_idx, SrcMDSpan src_mds, SubMDSpan sub_mds, std::index_sequence<SrcIdx...>, std::index_sequence<SubIdx...>, std::pair<int,int> p, SliceArgs ... slices) {
    using idx_t = typename SubMDSpan::index_type;
    return (sub_mds.extent(sub_idx)==static_cast<idx_t>(p.second-p.first)) && check_submdspan_match(++src_idx, ++sub_idx, src_mds, sub_mds, std::index_sequence<SrcIdx...,2>(), std::index_sequence<SubIdx...,1>(), slices...);
  }
  template<class SrcMDSpan, class SubMDSpan, size_t ... SrcIdx, size_t ... SubIdx, class ... SliceArgs>
  MDSPAN_INLINE_FUNCTION
  static bool check_submdspan_match(int src_idx, int sub_idx, SrcMDSpan src_mds, SubMDSpan sub_mds, std::index_sequence<SrcIdx...>, std::index_sequence<SubIdx...>, std::complex<double> p, SliceArgs ... slices) {
    using idx_t = typename SubMDSpan::index_type;
    return (sub_mds.extent(sub_idx)==static_cast<idx_t>(p.imag()-p.real())) && check_submdspan_match(++src_idx, ++sub_idx, src_mds, sub_mds, std::index_sequence<SrcIdx...,2>(), std::index_sequence<SubIdx...,1>(), slices...);
  }
  template<class SrcMDSpan, class SubMDSpan, size_t ... SrcIdx, size_t ... SubIdx, class ... SliceArgs>
  MDSPAN_INLINE_FUNCTION
  static bool check_submdspan_match(int src_idx, int sub_idx, SrcMDSpan src_mds, SubMDSpan sub_mds, std::index_sequence<SrcIdx...>, std::index_sequence<SubIdx...>,
                                     Kokkos::strided_slice<int,int,int> p, SliceArgs ... slices) {
    using idx_t = typename SubMDSpan::index_type;
    return (sub_mds.extent(sub_idx)==static_cast<idx_t>((p.extent+p.stride-1)/p.stride)) && check_submdspan_match(++src_idx, ++sub_idx, src_mds, sub_mds, std::index_sequence<SrcIdx...,3>(), std::index_sequence<SubIdx...,1>(), slices...);
  }
  template<class SrcMDSpan, class SubMDSpan, size_t ... SrcIdx, size_t ... SubIdx, class ... SliceArgs>
  MDSPAN_INLINE_FUNCTION
  static bool check_submdspan_match(int src_idx, int sub_idx, SrcMDSpan src_mds, SubMDSpan sub_mds, std::index_sequence<SrcIdx...>, std::index_sequence<SubIdx...>,
                                     Kokkos::strided_slice<int,std::integral_constant<int, 0>,std::integral_constant<int,0>>, SliceArgs ... slices) {
    return (sub_mds.extent(sub_idx)==0) && check_submdspan_match(++src_idx, ++sub_idx, src_mds, sub_mds, std::index_sequence<SrcIdx...,1>(), std::index_sequence<SubIdx...,0>(), slices...);
  }
  template<class SrcMDSpan, class SubMDSpan, size_t ... SrcIdx, size_t ... SubIdx, class ... SliceArgs>
  MDSPAN_INLINE_FUNCTION
  static bool check_submdspan_match(int src_idx, int sub_idx, SrcMDSpan src_mds, SubMDSpan sub_mds, std::index_sequence<SrcIdx...>, std::index_sequence<SubIdx...>, Kokkos::full_extent_t, SliceArgs ... slices) {
    return (sub_mds.extent(sub_idx)==src_mds.extent(src_idx)) && check_submdspan_match(++src_idx, ++sub_idx, src_mds, sub_mds, std::index_sequence<SrcIdx...,1>(), std::index_sequence<SubIdx...,1>(), slices...);
  }
  template<class SrcMDSpan, class SubMDSpan, size_t ... SrcIdx, size_t ... SubIdx>
  MDSPAN_INLINE_FUNCTION
  static bool check_submdspan_match(int, int, SrcMDSpan src_mds, SubMDSpan sub_mds, std::index_sequence<SrcIdx...>, std::index_sequence<SubIdx...>) {
#if MDSPAN_USE_BRACKET_OPERATOR
    if constexpr (SrcMDSpan::rank() == 0) {
      return (&src_mds[]==&sub_mds[]);
    } else if constexpr (SubMDSpan::rank() == 0) {
      return (&src_mds[SrcIdx...]==&sub_mds[]);
    } else {
      if(sub_mds.size() == 0) return true;
      return (&src_mds[SrcIdx...]==&sub_mds[SubIdx...]);
    }
#else
    if constexpr (SrcMDSpan::rank() == 0) {
      return (&src_mds()==&sub_mds());
    } else if constexpr (SubMDSpan::rank() == 0) {
      return (&src_mds(SrcIdx...)==&sub_mds());
    } else {
      if(sub_mds.size() == 0) return true;
      return (&src_mds(SrcIdx...)==&sub_mds(SubIdx...));
    }
#endif
  }

  static void run() {
    size_t* result = allocate_array<size_t>(1);
    int* data = allocate_array<int>(25000);
    map_t map{ typename mds_org_t::extents_type(ConstrArgs...) };
    mds_org_t src(data, map);

    dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
      auto sub = Kokkos::submdspan(src, create_slice_arg(SubArgs())...);
      bool match = check_submdspan_match(0, 0, src, sub, std::index_sequence<>(), std::index_sequence<>(), create_slice_arg(SubArgs())...);
      result[0] = match?1:0;
    });
    EXPECT_EQ(result[0], 1);
    free_array(data);
    free_array(result);
  }

};


TYPED_TEST_SUITE(TestSubMDSpan, submdspan_test_types);

TYPED_TEST(TestSubMDSpan, submdspan_return_type) {
  static_assert(std::is_same<typename TestFixture::mds_sub_t,
                             typename TestFixture::mds_sub_deduced_t>::value,
                "SubMDSpan: wrong return type");
  MDSPAN_IMPL_TESTS_RUN_TEST(TestFixture::run());
}

#ifdef MDSPAN_IMPL_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
TEST(TestSubmdspanIssue4060, Rank1) {
  auto x = std::array<int, 3>{};
  auto A = Kokkos::mdspan{x.data(), Kokkos::extents{3}};
  ASSERT_EQ(A.mapping().required_span_size(), 3);
  auto B = Kokkos::submdspan(A, std::tuple{3, 3});

  ASSERT_EQ(B.rank(), 1u);
  EXPECT_EQ(B.extent(0), 0);
  EXPECT_EQ(B.data_handle(), x.data() + A.mapping().required_span_size());
}

template<class MappingType>
void test_submdspan_issue4060_rank2_all(const MappingType& mapping)
{
  auto y = std::array<int, 9>{};
  ASSERT_EQ(mapping.extents().rank(), 2u);
  ASSERT_EQ(mapping.required_span_size(), y.size());
  auto C = Kokkos::mdspan{y.data(), mapping};
  auto D = Kokkos::submdspan(C, std::tuple{3u, 3u}, std::tuple{3u, 3u});

  ASSERT_EQ(D.rank(), 2u);
  EXPECT_EQ(D.extent(0), 0);
  EXPECT_EQ(D.extent(1), 0);
  EXPECT_EQ(D.data_handle(), y.data() + mapping.required_span_size());
}

TEST(TestSubmdspanIssue4060, Rank2_all) {
  Kokkos::dextents<size_t, 2> exts{3u, 3u};
  {
    using mapping_type = Kokkos::layout_left::mapping<Kokkos::dextents<size_t, 2>>;
    test_submdspan_issue4060_rank2_all(mapping_type{exts});
  }
  {
    using mapping_type = Kokkos::layout_right::mapping<Kokkos::dextents<size_t, 2>>;
    test_submdspan_issue4060_rank2_all(mapping_type{exts});
  }
  {
    using mapping_type = Kokkos::layout_stride::mapping<Kokkos::dextents<size_t, 2>>;
    std::array<size_t, 2> strides{1u, 3u};
    test_submdspan_issue4060_rank2_all(mapping_type{exts, strides});
  }
}

template<class MappingType>
void test_submdspan_issue4060_rank2_one(const MappingType& mapping)
{
  auto y = std::array<int, 9>{};
  ASSERT_EQ(mapping.extents().rank(), 2u);
  ASSERT_EQ(mapping.required_span_size(), y.size());
  auto C = Kokkos::mdspan{y.data(), mapping};
  auto D = Kokkos::submdspan(C, std::tuple{0u, 3u}, std::tuple{3u, 3u});

  ASSERT_EQ(D.rank(), 2u);
  EXPECT_EQ(D.extent(0), 3u);
  EXPECT_EQ(D.extent(1), 0);
  EXPECT_EQ(D.data_handle(), y.data() + mapping.required_span_size());
}

TEST(TestSubmdspanIssue4060, Rank2_one) {
  Kokkos::dextents<size_t, 2> exts{3u, 3u};
  {
    using mapping_type = Kokkos::layout_left::mapping<Kokkos::dextents<size_t, 2>>;
    test_submdspan_issue4060_rank2_one(mapping_type{exts});
  }
  {
    using mapping_type = Kokkos::layout_right::mapping<Kokkos::dextents<size_t, 2>>;
    test_submdspan_issue4060_rank2_one(mapping_type{exts});
  }
  {
    using mapping_type = Kokkos::layout_stride::mapping<Kokkos::dextents<size_t, 2>>;
    std::array<size_t, 2> strides{1u, 3u};
    test_submdspan_issue4060_rank2_one(mapping_type{exts, strides});
  }
}
#endif

// https://github.com/kokkos/mdspan/issues/362
TEST(TestSubmdspanIssue362, LayoutRightPadded) {
  float array[1024] = {};
  std::iota(array, array + 1024, 0.0f);
  // Original mdspan across a 4x5 data block.
  Kokkos::mdspan<float, Kokkos::extents<size_t, 4, 5>> m1(array);

  // The bottom-right 2x2 matrix
  auto m2 = submdspan(m1, std::pair(2, 4), std::pair(3, 5));
      
  EXPECT_EQ(m2.mapping().required_span_size(), m2.mapping()(1, 1) + 1);
}
