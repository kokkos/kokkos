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
#include <mdspan/mdarray.hpp>
#include <vector>

#include <gtest/gtest.h>


namespace KokkosEx = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

MDSPAN_IMPL_INLINE_VARIABLE constexpr auto dyn = Kokkos::dynamic_extent;

template<class MDSpan, class MDArray>
struct MDArrayToMDSpanOperatorTest {
  using mdspan_t = MDSpan;
  using c_mdspan_t = Kokkos::mdspan<const typename mdspan_t::element_type,
                                   typename mdspan_t::extents_type,
                                   typename mdspan_t::layout_type>;
  static void test_check(mdspan_t mds, MDArray& mda) {
    ASSERT_EQ(mds.data_handle(), mda.data());
    ASSERT_EQ(mds.extents(), mda.extents());
    ASSERT_EQ(mds.mapping(), mda.mapping());
  }
  static void test_check_const(c_mdspan_t mds, const MDArray& mda) {
    ASSERT_EQ(mds.data_handle(), mda.data());
    ASSERT_EQ(mds.extents(), mda.extents());
    ASSERT_EQ(mds.mapping(), mda.mapping());
  }
  template<class ... ConstrArgs>
  static void test(ConstrArgs ... args) {
    MDArray a(args...);
    test_check(a, a);
    test_check(a.to_mdspan(), a);
    const MDArray& c_a = a;
    test_check_const(c_a, a);
    test_check_const(c_a.to_mdspan(), a);
  }
};

TEST(TestMDArray,mdarray_to_mdspan) {
  MDArrayToMDSpanOperatorTest<Kokkos::mdspan <int, Kokkos::extents<int, dyn>>,
                              KokkosEx::mdarray<int, Kokkos::extents<int, dyn>>>::test(
                              100
                              );
  MDArrayToMDSpanOperatorTest<Kokkos::mdspan <int, Kokkos::extents<int, dyn>>,
                              KokkosEx::mdarray<int, Kokkos::extents<int, 100>>>::test(
                              100
                              );
}
