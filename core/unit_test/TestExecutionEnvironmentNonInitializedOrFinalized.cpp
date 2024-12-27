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

#include <cstdlib>
#include <type_traits>

#include "KokkosExecutionEnvironmentNeverInitializedFixture.hpp"

namespace {

using ExecutionEnvironmentNonInitializedOrFinalized_DeathTest =
    KokkosExecutionEnvironmentNeverInitialized;

struct NonTrivial {
  KOKKOS_FUNCTION NonTrivial() {}
};
static_assert(!std::is_trivial_v<NonTrivial>);

TEST_F(ExecutionEnvironmentNonInitializedOrFinalized_DeathTest,
       default_constructed_views) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  auto make_views = [] {
    Kokkos::View<int> v0;
    Kokkos::View<float*> v1;
    Kokkos::View<NonTrivial**> v2;
  };
  EXPECT_EXIT(
      {
        make_views();
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_EXIT(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        make_views();
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

}  // namespace
