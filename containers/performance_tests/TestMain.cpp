// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <cstdlib>

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
