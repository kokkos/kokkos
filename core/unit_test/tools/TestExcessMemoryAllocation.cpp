// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <vector>

constexpr uint64_t CHUNK =
    512ULL * 1024 * 1024 / sizeof(double);  // 512 MiB per chunk

void allocate_views(uint64_t chunk, uint64_t num_chunks) {
  Kokkos::initialize();
  {
    std::vector<Kokkos::View<double *, Kokkos::DefaultHostExecutionSpace>>
        views;
    for (int i = 0; i < num_chunks; ++i) {
      views.emplace_back("A", chunk);
    }
  }
  Kokkos::finalize();
}

TEST(ExcessMemoryAllocation_DeathTest, ExcessMemoryAllocationFailsTest) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP()
      << "Allocations > 4GB are not supported on 32-bit builds.";  // FIXME_32BIT
#endif
  // allocate 9 chunks ~4.5 GB
  ASSERT_EXIT(allocate_views(CHUNK, 9), ::testing::ExitedWithCode(1),
              ".*WARNING!.*Total allocation.*GB.*exceeds.*GB limit!");
}

TEST(ExcessMemoryAllocation, AllocationBelowThresholdAllowed) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP()
      << "Allocations > 4GB are not supported on 32-bit builds.";  // FIXME_32BIT
#endif
  // allocate 7 chunks ~3.5 GB
  allocate_views(CHUNK, 7);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
