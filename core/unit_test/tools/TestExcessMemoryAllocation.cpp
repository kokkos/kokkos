// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <vector>

void allocate_large_view() {
  Kokkos::initialize();
  {
    constexpr uint64_t CHUNK =
        512ULL * 1024 * 1024 / sizeof(double);  // 512 MiB per chunk
    constexpr int NUM_CHUNKS = 9;               // ~4.5 GiB total
    std::vector<Kokkos::View<double *, Kokkos::DefaultHostExecutionSpace>>
        views;
    for (int i = 0; i < NUM_CHUNKS; ++i) {
      views.emplace_back("A", CHUNK);
    }
  }
  Kokkos::finalize();
}

TEST(ExcessMemoryAllocationErrorsInTesting,
     ExcessMemoryAllocationErrorsInTesting) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP()
      << "Allocations > 4GB are not supported on 32-bit builds.";  // FIXME_32BIT
#endif
  ASSERT_EXIT(allocate_large_view(), ::testing::ExitedWithCode(1),
              ".*WARNING!.*Total allocation.*GB.*exceeds.*GB limit!");
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
