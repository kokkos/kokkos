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

#include "TestDeepCopy.hpp"
namespace {
struct Foo {
  KOKKOS_FUNCTION Foo() = default;
  KOKKOS_FUNCTION Foo(double) {
    Kokkos::abort("This should never be called");
  }
  KOKKOS_FUNCTION Foo& operator=(double val_) {
    val = val_;
    return *this;
  }
  float val;
  KOKKOS_FUNCTION bool operator==(double val_) {
    return val_ == static_cast<double>(val);
  }
};
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_0) {
  test_deep_copy_assignable_types<double, Foo>();
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_1) {
  test_deep_copy_assignable_types<double*, Foo*>(1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_2) {
  test_deep_copy_assignable_types<double**, Foo**>(1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_3) {
  test_deep_copy_assignable_types<double***, Foo***>(1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_4) {
  test_deep_copy_assignable_types<double****, Foo****>(1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_5) {
  test_deep_copy_assignable_types<double*****, Foo*****>(1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_6) {
  test_deep_copy_assignable_types<double******, Foo******>(1, 1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_7) {
  test_deep_copy_assignable_types<double*******, Foo*******>(1, 1, 1, 1, 1, 1,
                                                             1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_8) {
  test_deep_copy_assignable_types<double********, Foo********>(1, 1, 1, 1, 1, 1,
                                                               1, 1);
}
