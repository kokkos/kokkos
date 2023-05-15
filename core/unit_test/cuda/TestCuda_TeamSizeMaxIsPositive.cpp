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

#include <TestCuda_Category.hpp>
#include <Kokkos_Core.hpp>

namespace Test {
namespace Impl {

struct FunctorFor {
  KOKKOS_FUNCTION
  void operator()(
      Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type const&) const {}
};

struct TeamSizeTester {
  void run_test_team_size(Kokkos::AUTO_t teamSizeRequested,
                          Kokkos::AUTO_t vectorLengthRequested) {
    Kokkos::TeamPolicy<TEST_EXECSPACE> myTeam(10, teamSizeRequested,
                                              vectorLengthRequested);
    const auto maxTeamSize =
        myTeam.team_size_max(FunctorFor(), Kokkos::ParallelForTag());

    const auto recommendedTeamSize =
        myTeam.team_size_recommended(FunctorFor(), Kokkos::ParallelForTag());
    EXPECT_GT(maxTeamSize, 0);
  }

  void run_test_team_size(Kokkos::AUTO_t teamSizeRequested,
                          int vectorLengthRequested) {
    Kokkos::TeamPolicy<TEST_EXECSPACE> myTeam(10, teamSizeRequested,
                                              vectorLengthRequested);
    const auto maxTeamSize =
        myTeam.team_size_max(FunctorFor(), Kokkos::ParallelForTag());

    const auto recommendedTeamSize =
        myTeam.team_size_recommended(FunctorFor(), Kokkos::ParallelForTag());
    EXPECT_GT(maxTeamSize, 0);
  }

  void run_test_team_size(int teamSizeRequested,
                          Kokkos::AUTO_t vectorLengthRequested) {
    Kokkos::TeamPolicy<TEST_EXECSPACE> myTeam(10, teamSizeRequested,
                                              vectorLengthRequested);
    const auto maxTeamSize =
        myTeam.team_size_max(FunctorFor(), Kokkos::ParallelForTag());
    EXPECT_GT(maxTeamSize, 0);
  };

  void run_test_team_size(int teamSizeRequested, int vectorLengthRequested) {
    Kokkos::TeamPolicy<TEST_EXECSPACE> myTeam(10, teamSizeRequested,
                                              vectorLengthRequested);
    const auto maxTeamSize =
        myTeam.team_size_max(FunctorFor(), Kokkos::ParallelForTag());
    EXPECT_GT(maxTeamSize, 0);
  }

  void run_test_team_size(Kokkos::AUTO_t teamSizeRequested) {
    Kokkos::TeamPolicy<TEST_EXECSPACE> myTeam(10, teamSizeRequested);
    const auto maxTeamSize =
        myTeam.team_size_max(FunctorFor(), Kokkos::ParallelForTag());
    EXPECT_GT(maxTeamSize, 0);
  }

  void run_test_team_size(int teamSizeRequested) {
    Kokkos::TeamPolicy<TEST_EXECSPACE> myTeam(10, teamSizeRequested);
    const auto maxTeamSize =
        myTeam.team_size_max(FunctorFor(), Kokkos::ParallelForTag());
    EXPECT_GT(maxTeamSize, 0);
  }
};

}  // namespace Impl

TEST(cuda, testTeamSizeMaxIsPositive) {
  Impl::TeamSizeTester teamSizeTester;

  teamSizeTester.run_test_team_size(Kokkos::AUTO, Kokkos::AUTO);
  teamSizeTester.run_test_team_size(Kokkos::AUTO, 1);
  teamSizeTester.run_test_team_size(Kokkos::AUTO, 2);
  teamSizeTester.run_test_team_size(Kokkos::AUTO, 4);
  teamSizeTester.run_test_team_size(1, Kokkos::AUTO);
  teamSizeTester.run_test_team_size(2, Kokkos::AUTO);
  teamSizeTester.run_test_team_size(4, Kokkos::AUTO);
  teamSizeTester.run_test_team_size(1, 1);
  teamSizeTester.run_test_team_size(2, 2);
  teamSizeTester.run_test_team_size(4, 4);

  // Vector length not defined
  teamSizeTester.run_test_team_size(Kokkos::AUTO);
  teamSizeTester.run_test_team_size(1);
  teamSizeTester.run_test_team_size(2);
  teamSizeTester.run_test_team_size(4);
}

}  // namespace Test
