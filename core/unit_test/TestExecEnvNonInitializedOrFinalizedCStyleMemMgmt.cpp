//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "KokkosExecutionEnvironmentNeverInitializedFixture.hpp"

namespace {

using ExecutionEnvironmentNonInitializedOrFinalized_DeathTest =
    KokkosExecutionEnvironmentNeverInitialized;

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       c_style_memory_management) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  EXPECT_DEATH(
      { [[maybe_unused]] void* ptr = Kokkos::kokkos_malloc(1); },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_malloc\\(\\) \\*\\*before\\*\\* Kokkos::initialize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        [[maybe_unused]] void* ptr = Kokkos::kokkos_malloc(1);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_malloc\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        void* ptr = Kokkos::kokkos_malloc(1);
        Kokkos::finalize();
        Kokkos::kokkos_free(ptr);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_free\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        void* prev = Kokkos::kokkos_malloc(1);
        Kokkos::finalize();
        [[maybe_unused]] void* next = Kokkos::kokkos_realloc(prev, 2);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_realloc\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        // Take a fake pointer
        void* ptr = reinterpret_cast<void*>(0x8BADF00D);
        Kokkos::kokkos_free(ptr);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_free\\(\\) \\*\\*before\\*\\* Kokkos::initialize\\(\\) was "
      "called");
  EXPECT_DEATH(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        // Take a fake pointer
        void* ptr = reinterpret_cast<void*>(0xB105F00D);
        Kokkos::kokkos_free(ptr);
      },
      "Kokkos ERROR: attempting to perform C-style memory management via "
      "kokkos_free\\(\\) \\*\\*after\\*\\* Kokkos::finalize\\(\\) was called");
}

}  // namespace
