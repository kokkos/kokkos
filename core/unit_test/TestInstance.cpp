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

namespace Test {

/**
 * @test Ensure that any execution space fails in a consistent manner when @c Kokkos is not initialized.
 *
 * The expected behavior is that the execution space instance constructor aborts
 * with a message that look s like:
 *  @code
 *  Kokkos::<execution space name> : <label> : ERROR device not initialized
 *  @endcode
 */
TEST(TEST_CATEGORY, instance_verify_is_initialized) {
  using execution_space = Kokkos::DefaultExecutionSpace;

  ASSERT_FALSE(Kokkos::is_initialized());

  ASSERT_EXIT(
    execution_space space{},
    ::testing::KilledBySignal(SIGABRT),
    ::testing::ContainsRegex("Kokkos::[A-Za-z:]+ : [A-Za-z ]+ : ERROR device not initialized")
  );
}

} // namespace Test
