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

#include <cstdio>
#include <sstream>
#include <iostream>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace {

template <typename Policy>
void test_run_time_parameters() {
  int league_size = 131;
  int team_size   = 4 < Policy::execution_space::concurrency()
                      ? 4
                      : Policy::execution_space::concurrency();
#ifdef KOKKOS_ENABLE_HPX
  team_size = 1;
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (std::is_same<typename Policy::execution_space,
                   Kokkos::Experimental::OpenMPTarget>::value)
    team_size = 32;
#endif
  int chunk_size         = 4;
  int per_team_scratch   = 1024;
  int per_thread_scratch = 16;
  int scratch_size       = per_team_scratch + per_thread_scratch * team_size;

  Policy p1(league_size, team_size);
  ASSERT_EQ(p1.league_size(), league_size);
  ASSERT_EQ(p1.team_size(), team_size);
  ASSERT_GT(p1.chunk_size(), 0);
  ASSERT_EQ(size_t(p1.scratch_size(0)), 0u);

  Policy p2 = p1.set_chunk_size(chunk_size);
  ASSERT_EQ(p1.league_size(), league_size);
  ASSERT_EQ(p1.team_size(), team_size);
  ASSERT_EQ(p1.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p1.scratch_size(0)), 0u);

  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p2.scratch_size(0)), 0u);

  Policy p3 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(per_team_scratch));
  ASSERT_EQ(p3.league_size(), league_size);
  ASSERT_EQ(p3.team_size(), team_size);
  ASSERT_EQ(p3.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p3.scratch_size(0)), size_t(per_team_scratch));

  Policy p4 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(scratch_size));
  ASSERT_EQ(p4.league_size(), league_size);
  ASSERT_EQ(p4.team_size(), team_size);
  ASSERT_EQ(p4.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p4.scratch_size(0)), size_t(scratch_size));

  Policy p5 = p2.set_scratch_size(0, Kokkos::PerThread(per_thread_scratch),
                                  Kokkos::PerTeam(per_team_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(scratch_size));
  ASSERT_EQ(p5.league_size(), league_size);
  ASSERT_EQ(p5.team_size(), team_size);
  ASSERT_EQ(p5.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p5.scratch_size(0)), size_t(scratch_size));

  Policy p6 = p2.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch),
                                  Kokkos::PerThread(per_thread_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p2.scratch_size(0)), size_t(scratch_size));
  ASSERT_EQ(p6.league_size(), league_size);
  ASSERT_EQ(p6.team_size(), team_size);
  ASSERT_EQ(p6.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p6.scratch_size(0)), size_t(scratch_size));

  Policy p7 = p3.set_scratch_size(0, Kokkos::PerTeam(per_team_scratch),
                                  Kokkos::PerThread(per_thread_scratch));
  ASSERT_EQ(p3.league_size(), league_size);
  ASSERT_EQ(p3.team_size(), team_size);
  ASSERT_EQ(p3.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p3.scratch_size(0)), size_t(scratch_size));
  ASSERT_EQ(p7.league_size(), league_size);
  ASSERT_EQ(p7.team_size(), team_size);
  ASSERT_EQ(p7.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p7.scratch_size(0)), size_t(scratch_size));

  Policy p8;  // default constructed
  ASSERT_EQ(p8.league_size(), 0);
  ASSERT_EQ(size_t(p8.scratch_size(0)), 0u);
  p8 = p3;  // call assignment operator
  ASSERT_EQ(p3.league_size(), league_size);
  ASSERT_EQ(p3.team_size(), team_size);
  ASSERT_EQ(p3.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p3.scratch_size(0)), size_t(scratch_size));
  ASSERT_EQ(p8.league_size(), league_size);
  ASSERT_EQ(p8.team_size(), team_size);
  ASSERT_EQ(p8.chunk_size(), chunk_size);
  ASSERT_EQ(size_t(p8.scratch_size(0)), size_t(scratch_size));
}

TEST(TEST_CATEGORY, team_policy_runtime_parameters) {
  struct SomeTag {};

  using TestExecSpace   = TEST_EXECSPACE;
  using DynamicSchedule = Kokkos::Schedule<Kokkos::Dynamic>;
  using LongIndex       = Kokkos::IndexType<long>;

  // clang-format off
  test_run_time_parameters<Kokkos::TeamPolicy<TestExecSpace                                             >>();
  test_run_time_parameters<Kokkos::TeamPolicy<TestExecSpace,   DynamicSchedule, LongIndex               >>();
  test_run_time_parameters<Kokkos::TeamPolicy<LongIndex,       TestExecSpace,   DynamicSchedule         >>();
  test_run_time_parameters<Kokkos::TeamPolicy<DynamicSchedule, LongIndex,       TestExecSpace,   SomeTag>>();
  // clang-format on
}

}  // namespace
