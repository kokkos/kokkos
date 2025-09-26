// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

/**
 * Fixture that checks Kokkos is neither initialized nor finalized before and
 * after each test.
 */
class KokkosExecutionEnvironmentNeverInitialized : public ::testing::Test {
  static void checkNeverInitialized() {
    ASSERT_FALSE(Kokkos::is_initialized());
    ASSERT_FALSE(Kokkos::is_finalized());
  }

 protected:
  void SetUp() override { checkNeverInitialized(); }

  void TearDown() override { checkNeverInitialized(); }
};
