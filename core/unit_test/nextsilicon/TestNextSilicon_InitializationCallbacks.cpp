// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

int callback_ran = 0;

TEST(nextsilicon, InitializationCallbacksRun) { EXPECT_EQ(callback_ran, 1); }

}  // namespace

// FIXME_NEXTSILICON: integrate with existing InitializeFinalize tests once
// DeathTests are supported by NextSilicon toolchain.
int main(int argc, char* argv[]) {
  Kokkos::Impl::register_nextsilicon_initialization_callback(
      [] { ++callback_ran; });

  Kokkos::initialize(argc, argv);

  // Force linker to pull in Kokkos_NextSilicon.cpp so NextSilicon backend gets
  // registered via initialize_space_factory
  { Kokkos::Experimental::NextSilicon sp{}; }

  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
