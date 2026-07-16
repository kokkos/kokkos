// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

bool callback_ran        = false;
bool nested_callback_ran = false;

TEST(nextsilicon, InitializationCallbacksRun) { EXPECT_TRUE(callback_ran); }

TEST(nextsilicon, InitializationNestedCallbacksRun) {
  EXPECT_TRUE(nested_callback_ran);
}

// reset the callback state and check that the callback runs immediately
TEST(nextsilicon, InitializationCallbacksRunImmediately) {
  bool local_callback_ran = false;
  Kokkos::Impl::register_nextsilicon_initialization_callback(
      "TestNextSilicon_InitializationCallbacks::immediate",
      [&] { local_callback_ran = true; });
  EXPECT_TRUE(local_callback_ran);
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::Impl::register_nextsilicon_initialization_callback(
      "TestNextSilicon_InitializationCallbacks::deferred", [] {
        callback_ran = true;
        Kokkos::Impl::register_nextsilicon_initialization_callback(
            "TestNextSilicon_InitializationCallbacks::while_draining",
            [] { nested_callback_ran = true; });
      });

  Kokkos::initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
