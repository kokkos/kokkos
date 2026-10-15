// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <cstdlib>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char *argv[]) {
  // We want to use "threadsafe" by default while the default in GTest on Linux
  // is "fast"
  if (!std::getenv("GTEST_DEATH_TEST_STYLE"))
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
