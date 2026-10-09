// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  // Proposed in https://github.com/kokkos/kokkos/pull/9469
  if (!std::getenv("GTEST_DEATH_TEST_STYLE"))
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  ::testing::InitGoogleTest(&argc, argv);
  std::cout << GTEST_FLAG_GET(death_test_style) << '\n';
  return 0;
}
