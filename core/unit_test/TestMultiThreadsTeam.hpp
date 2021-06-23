/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_TEST_MultiThreadsTeam_HPP
#define KOKKOS_TEST_MultiThreadsTeam_HPP

#include <Kokkos_Core.hpp>

#include <thread>

namespace Test {
namespace {
template <typename ExecSpace>
struct TestMultiThreadsTeam {
  using value_type = int;

  using view_type = Kokkos::View<value_type *, ExecSpace>;

  struct VerifyInitTag {};
  struct ResetTag {};
  struct VerifyResetTag {};
  struct OffsetTag {};
  struct VerifyOffsetTag {};

  int N;
  int team_size;
  view_type results;
  view_type results_scan;

  TestMultiThreadsTeam(int const N_, int const team_size_)
      : N(N_),
        team_size(team_size_),
        results(Kokkos::view_alloc(Kokkos::WithoutInitializing, "results"), N) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(
      typename Kokkos::TeamPolicy<ExecSpace>::member_type const team) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size),
                         [&](int const i) {
                           int const j = team.league_rank() * team_size + i;
                           results(j)  = j;
                         });
  }

  void check_result_for() {
    auto results_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    for (int i = 0; i < N; ++i) ASSERT_EQ(results_host(i), i);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(
      typename Kokkos::TeamPolicy<ExecSpace>::member_type const team,
      value_type &update) const {
    value_type tmp = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, team_size),
        [&](int const i, value_type &innerUpdate) {
          int const j = team.league_rank() * team_size + i;
          innerUpdate += results(j);
        },
        tmp);

    if (team.team_rank() == 0) {
      update += tmp;
    }
  }
};
}  // namespace

TEST(TEST_CATEGORY, multiple_team_for) {
  int const N_0       = 100000;
  int const N_1       = 2 * N_0;
  int const team_size = 1;
  TestMultiThreadsTeam<TEST_EXECSPACE> range_0(N_0, team_size);
  TestMultiThreadsTeam<TEST_EXECSPACE> range_1(N_1, team_size);
  auto f_0 = [&]() {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<TEST_EXECSPACE>(N_0 / team_size, team_size),
        range_0);
  };
  auto f_1 = [&]() {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<TEST_EXECSPACE>(N_1 / team_size, team_size),
        range_1);
  };
  std::thread t_0(f_0);
  std::thread t_1(f_1);
  t_0.join();
  t_1.join();

  range_0.check_result_for();
  range_1.check_result_for();
}

TEST(TEST_CATEGORY, multiple_team_reduce) {
  size_t const N_0    = 10;
  size_t const N_1    = 2 * N_0;
  int total_0         = 0;
  int total_1         = 0;
  int const team_size = 1;
  int ref_0           = (N_0 - 1) * N_0 / 2;
  int ref_1           = (N_1 - 1) * N_1 / 2;
  TestMultiThreadsTeam<TEST_EXECSPACE> range_0(N_0 / team_size, team_size);
  TestMultiThreadsTeam<TEST_EXECSPACE> range_1(N_1 / team_size, team_size);
  auto f_0 = [&]() {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<TEST_EXECSPACE>(N_0 / team_size, team_size),
        range_0);
    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<TEST_EXECSPACE>(N_0 / team_size, team_size), range_0,
        total_0);
  };
  auto f_1 = [&]() {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<TEST_EXECSPACE>(N_1 / team_size, team_size),
        range_1);
    Kokkos::parallel_reduce(
        Kokkos::TeamPolicy<TEST_EXECSPACE>(N_1 / team_size, team_size), range_1,
        total_1);
  };
  std::thread t_0(f_0);
  std::thread t_1(f_1);
  t_0.join();
  t_1.join();

  ASSERT_EQ(total_0, ref_0);
  ASSERT_EQ(total_1, ref_1);
}
}  // namespace Test

#endif
