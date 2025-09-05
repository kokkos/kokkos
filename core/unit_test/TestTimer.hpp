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

#include <Kokkos_Timer.hpp>

#include <thread>
#include <type_traits>
#include <utility>

namespace {

TEST(TEST_CATEGORY, timer) {
  using namespace std::chrono_literals;

  Kokkos::Timer t;
  std::this_thread::sleep_for(5ms);
  auto elapsed = t.seconds();
  EXPECT_GE(elapsed, .005);
  EXPECT_LT(elapsed, 1.);

  std::this_thread::sleep_for(10ms);
  auto elapsed2 = std::as_const(t).seconds();
  EXPECT_GE(elapsed2, .015);
  EXPECT_GT(elapsed2, elapsed);

  t.reset();
  std::this_thread::sleep_for(5ms);
  auto elapsed3 = t.seconds();
  EXPECT_GE(elapsed3, .005);
  EXPECT_LT(elapsed3, elapsed2);
}

static_assert(!std::is_copy_constructible_v<Kokkos::Timer>);
static_assert(!std::is_move_constructible_v<Kokkos::Timer>);
static_assert(!std::is_copy_assignable_v<Kokkos::Timer>);
static_assert(!std::is_move_assignable_v<Kokkos::Timer>);

}  // namespace
