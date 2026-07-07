// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <alloca.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "gtest/gtest.h"

void allocate_large_view() {
  Kokkos::initialize();
  size_t very_large_size = 5 << 30;
  Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> a("A",very_large_size);
  Kokkos::finalize();
}

TEST(ExcessMemoryAllocationErrorsInTesting, ExcessMemoryAllocationErrorsInTesting){
    ASSERT_DEATH(allocate_large_view(), ".*WARNING!.*Total allocation.*GB.*exceeds.*GB limit!");
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}