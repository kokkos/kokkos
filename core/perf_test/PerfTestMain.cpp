// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <cstdlib>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <PerfTest_Category.hpp>

int main(int argc, char* argv[]) {
  // We want to use "threadsafe" by default while the default in GTest on Linux
  // is "fast"
  Kokkos::Impl::setenv("GTEST_DEATH_TEST_STYLE", "threadsafe",
                       /*overwrite=*/0);
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);

  (void)Test::command_line_num_args(argc);
  (void)Test::command_line_arg(0, argv);

  int result = RUN_ALL_TESTS();

  Kokkos::finalize();
  return result;
}
