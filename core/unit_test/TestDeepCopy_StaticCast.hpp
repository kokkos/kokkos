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
struct Bar {
  Bar() = default;
  KOKKOS_FUNCTION explicit Bar(double val_) : val(val_) {}
  double val;
  KOKKOS_FUNCTION bool operator==(double val_) {
    return val_ == static_cast<double>(val);
  }
};
}  // namespace

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_0) {
  test_deep_copy_assignable_types<double, Bar>();
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_1) {
  test_deep_copy_assignable_types<double*, Bar*>(1);
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_2) {
  test_deep_copy_assignable_types<double**, Bar**>(1, 1);
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_3) {
  test_deep_copy_assignable_types<double***, Bar***>(1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_4) {
  test_deep_copy_assignable_types<double****, Bar****>(1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_5) {
  test_deep_copy_assignable_types<double*****, Bar*****>(1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_6) {
  test_deep_copy_assignable_types<double******, Bar******>(1, 1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_7) {
  test_deep_copy_assignable_types<double*******, Bar*******>(1, 1, 1, 1, 1, 1,
                                                             1);
}

TEST(TEST_CATEGORY, deep_copy_non_assignable_types_rank_8) {
  test_deep_copy_assignable_types<double********, Bar********>(1, 1, 1, 1, 1, 1,
                                                               1, 1);
}
