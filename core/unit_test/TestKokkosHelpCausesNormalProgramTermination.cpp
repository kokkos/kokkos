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

import mykokkoscore;
#include <Kokkos_Macros.hpp>

#include "KokkosExecutionEnvironmentNeverInitializedFixture.hpp"

namespace {

using KokkosHelpCausesNormalProgramTermination_DeathTest =
    KokkosExecutionEnvironmentNeverInitialized;

TEST_F(KokkosHelpCausesNormalProgramTermination_DeathTest,
       print_help_and_exit_early) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  int argc = 1;

  char const *argv[] = {
      "--kokkos-help",
      nullptr,
  };

  ::testing::internal::CaptureStdout();

  EXPECT_EXIT(
      {
        Kokkos::initialize(argc, const_cast<char **>(argv));
        Kokkos::abort("better exit before getting there");
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");

  (void)::testing::internal::GetCapturedStdout();
}

}  // namespace
