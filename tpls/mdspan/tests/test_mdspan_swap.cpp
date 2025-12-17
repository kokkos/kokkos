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
#include "offload_utils.hpp"
#include "foo_customizations.hpp"


void test_mdspan_std_swap_static_extents() {
  size_t* errors = allocate_array<size_t>(1);
  errors[0] = 0;

  dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
    int data1[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int data2[12] = {21,22,23,24,25,26,27,28,29,30,31,32};
    Kokkos::mdspan<int, Kokkos::extents<size_t,3,4>> m1(data1);
    Kokkos::mdspan<int, Kokkos::extents<size_t,3,4>> m2(data2);
    Kokkos::extents<size_t,3,4> exts1;
    Kokkos::layout_right::mapping<Kokkos::extents<size_t, 3, 4>> map1(exts1);
    Kokkos::extents<size_t,3,4> exts2;
    Kokkos::layout_right::mapping<Kokkos::extents<size_t, 3, 4>> map2(exts2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.data_handle(), data1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.mapping(), map1);
    auto val1 = MDSPAN_IMPL_OP(m1,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val1, 1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.data_handle(), data2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.mapping(), map2);
    auto val2 = MDSPAN_IMPL_OP(m2,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val2, 21);
    swap(m1,m2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.data_handle(), data2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.mapping(), map2);
    val1 = MDSPAN_IMPL_OP(m1,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val1, 21);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.data_handle(), data1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.mapping(), map1);
    val2 = MDSPAN_IMPL_OP(m2,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val2, 1);
  });
  ASSERT_EQ(errors[0], 0);
  free_array(errors);
}

TEST(TestMDSpanSwap, std_swap_static_extents) {
  MDSPAN_IMPL_TESTS_RUN_TEST(test_mdspan_std_swap_static_extents())
}


void test_mdspan_std_swap_dynamic_extents() {
  size_t* errors = allocate_array<size_t>(1);
  errors[0] = 0;

  dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
    int data1[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int data2[12] = {21,22,23,24,25,26,27,28,29,30,31,32};
    Kokkos::mdspan<int, Kokkos::dextents<size_t,2>> m1(data1,3,4);
    Kokkos::mdspan<int, Kokkos::dextents<size_t,2>> m2(data2,4,3);
    Kokkos::dextents<size_t,2> exts1(3,4);
    Kokkos::layout_right::mapping<Kokkos::dextents<size_t,2>> map1(exts1);
    Kokkos::dextents<size_t,2> exts2(4,3);
    Kokkos::layout_right::mapping<Kokkos::dextents<size_t,2>> map2(exts2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.data_handle(), data1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.mapping(), map1);
    auto val1 = MDSPAN_IMPL_OP(m1,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val1, 1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.data_handle(), data2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.mapping(), map2);
    auto val2 = MDSPAN_IMPL_OP(m2,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val2, 21);
    swap(m1,m2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.data_handle(), data2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.mapping(), map2);
    val1 = MDSPAN_IMPL_OP(m1,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val1, 21);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.data_handle(), data1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.mapping(), map1);
    val2 = MDSPAN_IMPL_OP(m2,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val2, 1);
  });
  ASSERT_EQ(errors[0], 0);
  free_array(errors);
}

TEST(TestMDSpanSwap, std_swap_dynamic_extents) {
  MDSPAN_IMPL_TESTS_RUN_TEST(test_mdspan_std_swap_dynamic_extents())
}

// On HIP/CUDA we actually don't call through to swap via ADL
// so the foo swap test which has side effects will fail
#if !defined(MDSPAN_IMPL_HAS_HIP) && !defined(MDSPAN_IMPL_HAS_CUDA)
void test_mdspan_foo_swap_dynamic_extents() {
  size_t* errors = allocate_array<size_t>(1);
  errors[0] = 0;

  dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
    using map_t = Foo::layout_foo::template mapping<Kokkos::dextents<size_t ,2>>;
    using acc_t = Foo::foo_accessor<int>;

    int data1[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int data2[12] = {21,22,23,24,25,26,27,28,29,30,31,32};
    int flag1 = 9;
    int flag2 = 7;

    Kokkos::dextents<size_t, 2> exts1(3,4);
    map_t map1(exts1);
    acc_t acc1(&flag1);
    Kokkos::mdspan<int, Kokkos::dextents<size_t,2>,
                  Foo::layout_foo, acc_t> m1(Foo::foo_ptr<int>{data1}, map1, acc1);
    Kokkos::dextents<size_t, 2> exts2(4,3);
    map_t map2(exts2);
    acc_t acc2(&flag2);
    Kokkos::mdspan<int, Kokkos::dextents<size_t,2>,
                  Foo::layout_foo, acc_t> m2(Foo::foo_ptr<int>{data2}, map2, acc2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ((map1==map2), false);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.data_handle().data, data1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.mapping(), map1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.accessor().flag[0], 9);
    auto val1 = MDSPAN_IMPL_OP(m1,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val1, 1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.data_handle().data, data2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.mapping(), map2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.accessor().flag[0], 7);
    auto val2 = MDSPAN_IMPL_OP(m2,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val2, 21);
    swap(m1,m2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.data_handle().data, data2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.mapping(), map2);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m1.accessor().flag[0], 77);
    val1 = MDSPAN_IMPL_OP(m1,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val1, 21);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.data_handle().data, data1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.mapping(), map1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m2.accessor().flag[0], 99);
    val2 = MDSPAN_IMPL_OP(m2,0,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val2, 1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(flag1, 99);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(flag2, 77);
  });
  ASSERT_EQ(errors[0], 0);
  free_array(errors);
}

TEST(TestMDSpanSwap, foo_swap_dynamic_extents) {
  MDSPAN_IMPL_TESTS_RUN_TEST(test_mdspan_foo_swap_dynamic_extents())
}
#endif
