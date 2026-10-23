// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

int callback_ran          = 0;
int canceled_callback_ran = 0;

TEST(nextsilicon, InitializationCallbacksRun) { EXPECT_EQ(callback_ran, 1); }

TEST(nextsilicon, InitializationCallbacksCanBeCanceled) {
  EXPECT_EQ(canceled_callback_ran, 0);
}

TEST(nextsilicon, InitializationCallbacksCanOnlyBeRetrievedOnce) {
  EXPECT_FALSE(Kokkos::Impl::retrieve_nextsilicon_initialization_callback(0));
}

}  // namespace

// FIXME_NEXTSILICON: integrate with existing InitializeFinalize tests once
// DeathTests are supported by NextSilicon toolchain.
int main(int argc, char* argv[]) {
  Kokkos::Impl::register_nextsilicon_initialization_callback(
      [] { ++callback_ran; });
  auto canceled_handle =
      Kokkos::Impl::register_nextsilicon_initialization_callback(
          [] { ++canceled_callback_ran; });
  auto canceled_callback =
      Kokkos::Impl::retrieve_nextsilicon_initialization_callback(
          canceled_handle);
  if (!canceled_callback) {
    Kokkos::abort(
        "nextsilicon: initialization callback unexpectedly missing. Please "
        "report this.");
  }

  Kokkos::initialize(argc, argv);

  // Force linker to pull in Kokkos_NextSilicon.cpp so NextSilicon backend gets
  // registered via initialize_space_factory
  { Kokkos::Experimental::NextSilicon sp{}; }

  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
