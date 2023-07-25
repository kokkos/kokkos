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

namespace {

enum Enum { EZero, EOne };
enum EnumBool : bool { EBFalse, EBTrue };
enum class ScopedEnum { SEZero, SEOne };
enum class ScopedEnumShort : short { SESZero, SESOne };

TEST(TEST_CATEGORY, to_underlying) {
  using Kokkos::Impl::to_underlying;

  auto e0 = to_underlying(EZero);
  ASSERT_EQ(e0, 0);

  auto e1 = to_underlying(EOne);
  ASSERT_EQ(e1, 1);

  auto eb0 = to_underlying(EBFalse);
  bool b0  = false;
  ASSERT_EQ(eb0, b0);

  auto eb1 = to_underlying(EBTrue);
  bool b1  = true;
  ASSERT_EQ(eb1, b1);

  auto se0 = to_underlying(ScopedEnum::SEZero);
  ASSERT_EQ(se0, 0);

  auto se1 = to_underlying(ScopedEnum::SEOne);
  ASSERT_EQ(se1, 1);

  auto ses0 = to_underlying(ScopedEnumShort::SESZero);
  short s0  = 0;
  ASSERT_EQ(ses0, s0);

  auto ses1 = to_underlying(ScopedEnumShort::SESOne);
  short s1  = 1;
  ASSERT_EQ(ses1, s1);
}

}  // namespace
