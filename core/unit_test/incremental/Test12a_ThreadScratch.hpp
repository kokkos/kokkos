/*
//@HEADER
// ************************************************************************
//
// Kokkos v. 2.0
// Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

// @Kokkos_Feature_Level_Required:6
// Unit test for hierarchical parallelism
// Create concurrent work hierarchically and verify if
// contributions of paticipating processing units corresponds to expected value
// Use a scratch pad memory for each team
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

// Degrees of concurrency per nesting level
#define N 16
#define M 16

namespace Test {

template <class ExecSpace>
struct ThreadScratch {
  void run() {
    typedef Kokkos::TeamPolicy<ExecSpace> teamPolicy;
    typedef typename Kokkos::TeamPolicy<ExecSpace>::member_type memberType;
    typedef Kokkos::View<int **, ExecSpace> dataType;
    typedef Kokkos::View<int *, ExecSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        scratchDataType;
    int scratchSize = scratchDataType::shmem_size(M);

    dataType v("Matrix", N, M);

    Kokkos::parallel_for(
        "Team",
        teamPolicy(N, M).set_scratch_size(0, Kokkos::PerThread(scratchSize)),
        KOKKOS_LAMBDA(const memberType &team) {
          // Allocate and use scratch pad memory
          scratchDataType v_S(team.thread_scratch(0), M);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, M),
                               [&](const int m) { v_S(m) = 0xABC; });
          team.team_barrier();
          // Aggregate contributions
          const int n = team.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M),
                               [&](const int m) { v(n, m) = v_S(m); });
        });

    Kokkos::fence();
    auto v_H = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);

    int check = 0;
    for (int n = 0; n < N; ++n)
      for (int m = 0; m < M; ++m) check += ((v_H(n, m) ^ 0xABC) == 0) ? 0 : 1;
    ASSERT_EQ(check, 0);
  }
};

TEST(TEST_CATEGORY, ThreadScratch) {
  ThreadScratch<TEST_EXECSPACE> test;
  test.run();
}

}  // namespace Test