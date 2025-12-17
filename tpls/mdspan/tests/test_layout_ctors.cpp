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

template <class> struct TestLayoutCtors;
template <class Mapping, size_t... DynamicSizes>
struct TestLayoutCtors<std::tuple<
  Mapping,
  std::integer_sequence<size_t, DynamicSizes...>
>> : public ::testing::Test {
  using mapping_type = Mapping;
  using extents_type = typename mapping_type::extents_type;
  Mapping map = { extents_type{ DynamicSizes... } };
};

template <class Extents, size_t... DynamicSizes>
using test_left_type = std::tuple<
  typename Kokkos::layout_left::template mapping<Extents>,
  std::integer_sequence<size_t, DynamicSizes...>
>;

template <class Extents, size_t... DynamicSizes>
using test_right_type = std::tuple<
  typename Kokkos::layout_right::template mapping<Extents>,
  std::integer_sequence<size_t, DynamicSizes...>
>;

using layout_test_types =
  ::testing::Types<
    test_left_type<Kokkos::extents<size_t,10>>,
    test_right_type<Kokkos::extents<size_t,10>>,
    //----------
    test_left_type<Kokkos::extents<size_t,dyn>, 10>,
    test_right_type<Kokkos::extents<size_t,dyn>, 10>,
    //----------
    test_left_type<Kokkos::extents<size_t,dyn, 10>, 5>,
    test_left_type<Kokkos::extents<size_t,5, dyn>, 10>,
    test_left_type<Kokkos::extents<size_t,5, 10>>,
    test_right_type<Kokkos::extents<size_t,dyn, 10>, 5>,
    test_right_type<Kokkos::extents<size_t,5, dyn>, 10>,
    test_right_type<Kokkos::extents<size_t,5, 10>>
  >;

TYPED_TEST_SUITE(TestLayoutCtors, layout_test_types);

TYPED_TEST(TestLayoutCtors, default_ctor) {
  // Default constructor ensures extents() == Extents() is true.
  auto m = typename TestFixture::mapping_type();
  ASSERT_EQ(m.extents(), typename TestFixture::extents_type());
  auto m2 = typename TestFixture::mapping_type{};
  ASSERT_EQ(m2.extents(), typename TestFixture::extents_type{});
  ASSERT_EQ(m, m2);
  ASSERT_FALSE(m != m2);
}

template <class> struct TestLayoutCompatCtors;
template <class Mapping, size_t... DynamicSizes, class Mapping2, size_t... DynamicSizes2>
struct TestLayoutCompatCtors<std::tuple<
  Mapping,
  std::integer_sequence<size_t, DynamicSizes...>,
  Mapping2,
  std::integer_sequence<size_t, DynamicSizes2...>
>> : public ::testing::Test {
  using mapping_type1 = Mapping;
  using mapping_type2 = Mapping2;
  using extents_type1 = std::remove_reference_t<decltype(std::declval<mapping_type1>().extents())>;
  using extents_type2 = std::remove_reference_t<decltype(std::declval<mapping_type2>().extents())>;
  Mapping map1 = { extents_type1{ DynamicSizes... } };
  Mapping2 map2 = { extents_type2{ DynamicSizes2... } };
};

template <class E1, class S1, class E2, class S2>
using test_left_type_compatible = std::tuple<
  typename Kokkos::layout_left::template mapping<E1>, S1,
  typename Kokkos::layout_left::template mapping<E2>, S2
>;
template <class E1, class S1, class E2, class S2>
using test_right_type_compatible = std::tuple<
  typename Kokkos::layout_right::template mapping<E1>, S1,
  typename Kokkos::layout_right::template mapping<E2>, S2
>;
template <size_t... Ds>
using sizes = std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using exts = Kokkos::extents<size_t,Ds...>;

template <template <class, class, class, class> class test_case_type>
using compatible_layout_test_types =
  ::testing::Types<
    test_case_type<exts<dyn>, sizes<10>, exts<10>, sizes<>>,
    //--------------------
    test_case_type<exts<dyn, 10>, sizes<5>, exts<5, dyn>, sizes<10>>,
    test_case_type<exts<dyn, dyn>, sizes<5, 10>, exts<5, dyn>, sizes<10>>,
    test_case_type<exts<dyn, dyn>, sizes<5, 10>, exts<dyn, 10>, sizes<5>>,
    test_case_type<exts<dyn, dyn>, sizes<5, 10>, exts<5, 10>, sizes<>>,
    test_case_type<exts<5, 10>, sizes<>, exts<5, dyn>, sizes<10>>,
    test_case_type<exts<5, 10>, sizes<>, exts<dyn, 10>, sizes<5>>,
    //--------------------
    test_case_type<exts<dyn, dyn, 15>, sizes<5, 10>, exts<5, dyn, 15>, sizes<10>>,
    test_case_type<exts<5, 10, 15>, sizes<>, exts<5, dyn, 15>, sizes<10>>,
    test_case_type<exts<5, 10, 15>, sizes<>, exts<dyn, dyn, dyn>, sizes<5, 10, 15>>
  >;

using left_compatible_test_types = compatible_layout_test_types<test_left_type_compatible>;
using right_compatible_test_types = compatible_layout_test_types<test_right_type_compatible>;

template <class T> struct TestLayoutLeftCompatCtors : TestLayoutCompatCtors<T> { };
template <class T> struct TestLayoutRightCompatCtors : TestLayoutCompatCtors<T> { };

TYPED_TEST_SUITE(TestLayoutLeftCompatCtors, left_compatible_test_types);
TYPED_TEST_SUITE(TestLayoutRightCompatCtors, right_compatible_test_types);

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_construct_1) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(m1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_construct_1) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(m1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_construct_2) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(m2.extents(), this->map1.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_construct_2) {
  // The compatible mapping constructor ensures extents() == other.extents() is true.
  auto m2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(m2.extents(), this->map1.extents());
}

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_assign_1) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type2, typename TestFixture::mapping_type1>)
    this->map1 = this->map2;
  else
  #endif
    this->map1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_assign_1) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type2, typename TestFixture::mapping_type1>)
    this->map1 = this->map2;
  else
  #endif
    this->map1 = typename TestFixture::mapping_type1(this->map2);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutLeftCompatCtors, compatible_assign_2) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type1, typename TestFixture::mapping_type2>)
    this->map2 = this->map1;
  else
  #endif
    this->map2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TYPED_TEST(TestLayoutRightCompatCtors, compatible_assign_2) {
  #if MDSPAN_HAS_CXX_17
  if constexpr (std::is_convertible_v<typename TestFixture::mapping_type1, typename TestFixture::mapping_type2>)
    this->map2 = this->map1;
  else
  #endif
    this->map2 = typename TestFixture::mapping_type2(this->map1);
  ASSERT_EQ(this->map1.extents(), this->map2.extents());
}

TEST(TestLayoutLeftListInitialization, test_layout_left_extent_initialization) {
  Kokkos::layout_left::mapping<Kokkos::extents<size_t,dyn, dyn>> m{Kokkos::dextents<size_t,2>{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 16);
  ASSERT_TRUE(m.is_exhaustive());
}

// FIXME: CUDA NVCC including 12.0 does not like CTAD on nested classes
#if defined(MDSPAN_IMPL_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION) && !defined(__NVCC__)
TEST(TestLayoutLeftCTAD, test_layout_left_ctad) {
  Kokkos::layout_left::mapping m{Kokkos::extents{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 16);
  ASSERT_TRUE(m.is_exhaustive());
}
#endif

TEST(TestLayoutRightListInitialization, test_layout_right_extent_initialization) {
  Kokkos::layout_right::mapping<Kokkos::extents<size_t,dyn, dyn>> m{Kokkos::dextents<size_t,2>{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 32);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_exhaustive());
}

// FIXME: CUDA NVCC including 12.0 does not like CTAD on nested classes
#if defined(MDSPAN_IMPL_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION) && !defined(__NVCC__)
TEST(TestLayoutRightCTAD, test_layout_right_ctad) {
  Kokkos::layout_right::mapping m{Kokkos::extents{16, 32}};
  ASSERT_EQ(m.extents().rank(), 2);
  ASSERT_EQ(m.extents().rank_dynamic(), 2);
  ASSERT_EQ(m.extents().extent(0), 16);
  ASSERT_EQ(m.extents().extent(1), 32);
  ASSERT_EQ(m.stride(0), 32);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_exhaustive());
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

TEST(TestLayoutLeftStrideConstraint, test_layout_left_stride_constraint) {
  Kokkos::extents<int,16> ext1d{};
  Kokkos::layout_left::mapping m1d{ext1d};
  ASSERT_EQ (m1d.extents(), ext1d);
  ASSERT_TRUE ((is_stride_avail< decltype(m1d), int >::value));

  Kokkos::extents<int> ext0d{};
  Kokkos::layout_left::mapping m0d{ext0d};
  ASSERT_EQ (m0d.extents(), ext0d);
  ASSERT_FALSE((is_stride_avail< decltype(m0d), int >::value));
}

TEST(TestLayoutRightStrideConstraint, test_layout_right_stride_constraint) {
  Kokkos::extents<int,16> ext1d{};
  Kokkos::layout_right::mapping m1d{ext1d};
  ASSERT_EQ (m1d.extents(), ext1d);
  ASSERT_TRUE ((is_stride_avail< decltype(m1d), int >::value));

  Kokkos::extents<int> ext0d{};
  Kokkos::layout_right::mapping m0d{ext0d};
  ASSERT_EQ (m0d.extents(), ext0d);
  ASSERT_FALSE((is_stride_avail< decltype(m0d), int >::value));
}
#endif
