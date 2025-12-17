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
#include <type_traits>
#include <gtest/gtest.h>

namespace test {

template<class Extents>
static constexpr bool is_extents_v = false;

template<class IndexType, std::size_t ... Exts>
static constexpr bool is_extents_v<
  MDSPAN_IMPL_STANDARD_NAMESPACE :: extents<IndexType, Exts...>
> = true;

template<std::size_t Rank>
void test_dims_with_one_template_argument()
{
  using d = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE :: dims<Rank>;
  static_assert(test::is_extents_v<d>, "dims<Rank> is not an extents specialization");
  static_assert(std::is_same<typename d::index_type, std::size_t>::value, "dims::index_type is wrong");
  static_assert(d::rank() == Rank, "dims::rank() is wrong");
}

template<std::size_t Rank, class ExpectedIndexType>
void test_dims_with_two_template_arguments()
{
  using d = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE :: dims<Rank, ExpectedIndexType>;
  static_assert(test::is_extents_v<d>, "dims<Rank,T> is not an extents specialization");
  static_assert(std::is_same<typename d::index_type, ExpectedIndexType>::value, "dims::index_type is wrong");
  static_assert(d::rank() == Rank, "dims::rank() is wrong");
}
  
} // namespace test

TEST(TestDims, Test0)
{
  using test::test_dims_with_one_template_argument;
  using test::test_dims_with_two_template_arguments;

  test_dims_with_one_template_argument<0>();
  test_dims_with_one_template_argument<1>();
  test_dims_with_one_template_argument<2>();
  test_dims_with_one_template_argument<3>();
  test_dims_with_one_template_argument<4>();
  test_dims_with_one_template_argument<5>();
  test_dims_with_one_template_argument<6>();
  test_dims_with_one_template_argument<7>();
  test_dims_with_one_template_argument<8>();

  test_dims_with_two_template_arguments<0, std::size_t>();
  test_dims_with_two_template_arguments<1, std::size_t>();
  test_dims_with_two_template_arguments<2, std::size_t>();
  test_dims_with_two_template_arguments<3, std::size_t>();
  test_dims_with_two_template_arguments<4, std::size_t>();
  test_dims_with_two_template_arguments<5, std::size_t>();
  test_dims_with_two_template_arguments<6, std::size_t>();
  test_dims_with_two_template_arguments<7, std::size_t>();
  test_dims_with_two_template_arguments<8, std::size_t>();
}
