// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

bool callback_ran = false;

TEST(nextsilicon, InitializationCallbacksRun) { EXPECT_TRUE(callback_ran); }

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::Impl::register_nextsilicon_initialization_callback(
      "TestNextSilicon_InitializationCallbacks::deferred",
      [] { callback_ran = true; });

  Kokkos::initialize(argc, argv);

  // Force linker to pull in Kokkos_NextSilicon.cpp so NextSilicon backend get
  // registered via initialize_space_factory
  { Kokkos::Experimental::NextSilicon sp{}; }

  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
