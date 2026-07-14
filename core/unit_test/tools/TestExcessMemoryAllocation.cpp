// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstdint>

void allocate_large_view() {
  Kokkos::initialize();
  {
    uint64_t very_large_size = (5 << 30);
    Kokkos::View<double *, Kokkos::DefaultHostExecutionSpace> a(
        "A", very_large_size);
  }
  Kokkos::finalize();
}

TEST(ExcessMemoryAllocationErrorsInTesting,
     ExcessMemoryAllocationErrorsInTesting) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP()
      << "Allocations > 4GB are not supported on 32-bit builds.";  // FIXME_32BIT
#endif
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
