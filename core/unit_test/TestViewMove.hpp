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
#ifndef TESTVIEWMOVE_HPP_
#define TESTVIEWMOVE_HPP_

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace Test {

template <typename view_t>
struct TestViewMove {
  template <typename T, typename = std::enable_if_t<
                            std::is_same_v<std::decay_t<T>, view_t>>>
  TestViewMove(T&& view_) : view(std::forward<T>(view_)) {}

  view_t view;
};

/**
 * @test Ensure that @ref Kokkos::View and its members have proper move
 *       semantics.
 */
TEST(TEST_CATEGORY, view_move) {
  using execution_space = TEST_EXECSPACE;
  using view_t          = Kokkos::View<double*, execution_space>;
  using tester_t        = TestViewMove<view_t>;

  view_t view("view move test", 5);

  EXPECT_EQ(view.use_count(), 1);
  EXPECT_TRUE(view.is_allocated());

  tester_t tester{std::move(view)};

  //! As the view was moved, it's left in a pristine (a.k.a. *empty*) state.
  EXPECT_EQ(view.use_count(), 0);
  EXPECT_FALSE(view.is_allocated());

  EXPECT_EQ(tester.view.use_count(), 1);
  EXPECT_TRUE(tester.view.is_allocated());
}

}  // namespace Test

#endif  // TESTVIEWMOVE_HPP_
