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

#include <gtest/gtest.h>

MDSPAN_IMPL_INLINE_VARIABLE constexpr auto dyn = Kokkos::dynamic_extent;

template <class> struct TestLayoutStride;
template <size_t... Extents, size_t... DynamicSizes, size_t... StaticStrides, size_t... DynamicStrides>
struct TestLayoutStride<std::tuple<
  Kokkos::extents<size_t,Extents...>,
  std::integer_sequence<size_t, DynamicSizes...>,
  std::integer_sequence<size_t, StaticStrides...>,
  std::integer_sequence<size_t, DynamicStrides...>
>> : public ::testing::Test {
  using extents_type = Kokkos::extents<size_t,Extents...>;
  using mapping_type = typename Kokkos::layout_stride::template mapping<extents_type>;
  mapping_type map = { extents_type{ DynamicSizes... }, std::array<size_t, sizeof...(DynamicStrides)>{ DynamicStrides... } };
};

template <size_t... Extents>
using exts = Kokkos::extents<size_t,Extents...>;
template <size_t... Vals>
using ints = std::integer_sequence<size_t, Vals...>;
template <class E, class DSz, class SStr, class DStr>
using layout_stride_case_t =
  std::tuple<E, DSz, SStr, DStr>;

using extents_345_t = Kokkos::extents<size_t,3, 4, 5>;
using extents_3dyn5_t = Kokkos::extents<size_t,3, dyn, 5>;
using extents_ddd_t = Kokkos::extents<size_t,dyn, dyn, dyn>;
using zero_stride_maps =
  ::testing::Types<
    layout_stride_case_t<extents_345_t, ints<>, ints<dyn, dyn, dyn>, ints<0, 0, 0>>,
    layout_stride_case_t<extents_3dyn5_t, ints<4>, ints<dyn, dyn, dyn>, ints<0, 0, 0>>,
    layout_stride_case_t<extents_ddd_t, ints<3, 4, 5>, ints<dyn, dyn, dyn>, ints<0, 0, 0>>
  >;

template <class T>
struct TestLayoutStrideAllZero : TestLayoutStride<T> { };

TYPED_TEST_SUITE(TestLayoutStrideAllZero, zero_stride_maps);

TYPED_TEST(TestLayoutStrideAllZero, test_required_span_size) {
  ASSERT_EQ(this->map.required_span_size(), 1);
}

TYPED_TEST(TestLayoutStrideAllZero, test_mapping) {
  using index_type = decltype(this->map.extents().extent(0));
  for(index_type i = 0; i < this->map.extents().extent(0); ++i) {
    for(index_type j = 0; j < this->map.extents().extent(1); ++j) {
      for (index_type k = 0; k < this->map.extents().extent(2); ++k) {
        ASSERT_EQ(this->map(i, j, k), 0);
      }
    }
  }
}

#ifdef __cpp_lib_span
TEST(TestLayoutStrideSpanConstruction, test_from_span_construction) {
  using map_t = Kokkos::layout_stride::mapping<Kokkos::extents<size_t,dyn, dyn>>;
  std::array<int,2> strides{1,128};
  map_t m1(Kokkos::dextents<size_t,2>{16, 32}, std::span<int,2>{strides.data(),2});
  // The following is actually not supported by layout_stride in the standard?
  //map_t m2(Kokkos::dextents<size_t,2>{16, 32}, std::span<int,std::dynamic_extent>{strides.data(),2});
  //ASSERT_EQ(m1, m2);
  ASSERT_EQ(m1.extents().rank(), 2);
  ASSERT_EQ(m1.extents().rank_dynamic(), 2);
  ASSERT_EQ(m1.extents().extent(0), 16);
  ASSERT_EQ(m1.extents().extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 128);
  ASSERT_EQ(m1.strides()[0], 1);
  ASSERT_EQ(m1.strides()[1], 128);
  ASSERT_EQ(m1.required_span_size(),1+15*1+31*128);
  ASSERT_FALSE(m1.is_exhaustive());
}
#endif

TEST(TestLayoutStrideListInitialization, test_list_initialization) {
  Kokkos::layout_stride::mapping<Kokkos::extents<size_t,dyn, dyn>> m{Kokkos::dextents<size_t,2>{16, 32}, std::array<int,2>{1, 128}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 128);
  ASSERT_EQ(m.strides()[0], 1);
  ASSERT_EQ(m.strides()[1], 128);
  ASSERT_EQ(m.required_span_size(),1+15*1+31*128);
  ASSERT_FALSE(m.is_exhaustive());
}

template <class> struct TestLayoutEquality;
template <class Mapping, size_t... DynamicSizes, class Mapping2, size_t... DynamicSizes2, class Equality>
struct TestLayoutEquality<std::tuple<
  Mapping,
  std::integer_sequence<size_t, DynamicSizes...>,
  Mapping2,
  std::integer_sequence<size_t, DynamicSizes2...>,
  Equality
>> : public ::testing::Test {
  using mapping_type1 = Mapping;
  using mapping_type2 = Mapping2;
  using extents_type1 = std::remove_reference_t<decltype(std::declval<mapping_type1>().extents())>;
  using extents_type2 = std::remove_reference_t<decltype(std::declval<mapping_type2>().extents())>;
  mapping_type1 map1 = { extents_type1{}, std::array<size_t, sizeof...(DynamicSizes)>{ DynamicSizes... } };
  mapping_type2 map2 = { extents_type2{}, std::array<size_t, sizeof...(DynamicSizes2)>{ DynamicSizes2... } };
  bool equal = Equality::value;
};

template <class E1, class S1, class E2, class S2, class Equal>
using test_stride_equality = std::tuple<
  typename Kokkos::layout_stride::template mapping<E1>, S1,
  typename Kokkos::layout_stride::template mapping<E2>, S2,
  Equal
>;
template <size_t... Ds>
using sizes = std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using exts = Kokkos::extents<size_t,Ds...>;

template <template <class, class, class, class, class> class test_case_type>
using equality_test_types =
  ::testing::Types<
    test_case_type<exts<16, 32>, sizes<1, 16>, exts<16, 32>, sizes<1, 16>, std:: true_type>,
    test_case_type<exts<16, 32>, sizes<1, 16>, exts<16, 32>, sizes<1, 17>, std::false_type>,
    test_case_type<exts<16, 32>, sizes<1, 16>, exts<16, 64>, sizes<1, 16>, std::false_type>,
    test_case_type<exts<16, 32>, sizes<1, 16>, exts<16, 64>, sizes<1, 17>, std::false_type>
  >;

using layout_stride_equality_test_types = equality_test_types<test_stride_equality>;

TYPED_TEST_SUITE(TestLayoutEquality, layout_stride_equality_test_types);

TYPED_TEST(TestLayoutEquality, equality_op) {
  ASSERT_EQ(this->map1 == this->map2, this->equal);
}

// FIXME: CUDA NVCC including 12.0 does not like CTAD on nested classes
#if defined(MDSPAN_IMPL_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION) && !defined(__NVCC__)
TEST(TestLayoutStrideCTAD, test_ctad) {
  // This is not possible wiht the array constructor we actually provide
  /*
  Kokkos::layout_stride::mapping m0{Kokkos::extents{16, 32}, Kokkos::extents{1, 128}};
  ASSERT_EQ(m0.extents().rank(), 2);
  ASSERT_EQ(m0.extents().rank_dynamic(), 2);
  ASSERT_EQ(m0.extents().extent(0), 16);
  ASSERT_EQ(m0.extents().extent(1), 32);
  ASSERT_EQ(m0.stride(0), 1);
  ASSERT_EQ(m0.stride(1), 128);
  ASSERT_EQ(m0.strides(), (std::array<std::size_t, 2>{1, 128}));
  ASSERT_FALSE(m0.is_exhaustive());
  */
  Kokkos::layout_stride::mapping m1{Kokkos::extents{16, 32}, std::array{1, 128}};
  ASSERT_EQ(m1.extents().rank(), 2);
  ASSERT_EQ(m1.extents().rank_dynamic(), 2);
  ASSERT_EQ(m1.extents().extent(0), 16);
  ASSERT_EQ(m1.extents().extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 128);
  ASSERT_EQ(m1.strides()[0], 1);
  ASSERT_EQ(m1.strides()[1], 128);
  ASSERT_EQ(m1.required_span_size(),1+15*1+31*128);
  ASSERT_FALSE(m1.is_exhaustive());

// TODO These won't work with our current implementation, because the array will
// be deduced as the extent type, leading to a `static_assert`. We can probably
// get around this with a clever refactoring, or by using an alias template for
// `mapping` and deduction guides.
/*
  Kokkos::layout_stride::mapping m2{std::array{16, 32}, Kokkos::extents{1, 128}};
  ASSERT_EQ(m2.extents().rank(), 2);
  ASSERT_EQ(m2.extents().rank_dynamic(), 2);
  ASSERT_EQ(m2.extents().extent(0), 16);
  ASSERT_EQ(m2.extents().extent(1), 32);
  ASSERT_EQ(m2.stride(0), 1);
  ASSERT_EQ(m2.stride(1), 128);
  ASSERT_FALSE(m2.is_exhaustive());

  Kokkos::layout_stride::mapping m3{std::array{16, 32}, std::array{1, 128}};
  ASSERT_EQ(m3.extents().rank(), 2);
  ASSERT_EQ(m3.extents().rank_dynamic(), 2);
  ASSERT_EQ(m3.extents().extent(0), 16);
  ASSERT_EQ(m3.extents().extent(1), 32);
  ASSERT_EQ(m3.stride(0), 1);
  ASSERT_EQ(m3.stride(1), 128);
  ASSERT_FALSE(m3.is_exhaustive());
*/
}
#endif

#if MDSPAN_HAS_CXX_20
template< class T, class RankType, class = void >
struct is_stride_avail : std::false_type {};

template< class T, class RankType >
struct is_stride_avail< T
                      , RankType
                      , std::enable_if_t< std::is_same< decltype( std::declval<T>().stride( std::declval<RankType>() ) )
                                                      , typename T::index_type
                                                      >::value
                                        >
                      > : std::true_type {};

TEST(TestLayoutStrideStrideConstraint, test_layout_stride_stride_constraint) {
  Kokkos::extents<int,16> ext1d{};
  Kokkos::layout_stride::mapping m1d{ext1d, std::array<int,1>{1}};
  ASSERT_EQ (m1d.extents(), ext1d);
  ASSERT_TRUE ((is_stride_avail< decltype(m1d), int >::value));

  Kokkos::extents<int> ext0d{};
  Kokkos::layout_stride::mapping m0d{ext0d, std::array<int,0>{}};
  // This is weird, but its what the standard says
  ASSERT_EQ (m0d.extents(), ext0d);
  ASSERT_TRUE((is_stride_avail< decltype(m0d), int >::value));
}
#endif
