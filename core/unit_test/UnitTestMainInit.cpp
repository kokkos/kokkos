// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <cstdlib>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  // We want to use "threadsafe" by default while the default in GTest on Linux
  // is "fast"
  if (!std::getenv("GTEST_DEATH_TEST_STYLE"))
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
