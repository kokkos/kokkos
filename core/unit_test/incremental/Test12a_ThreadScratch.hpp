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

// @Kokkos_Feature_Level_Required:12
// Unit test for hierarchical parallelism
// Create concurrent work hierarchically and verify if
// contributions of paticipating processing units corresponds to expected value
// Use a scratch pad memory for each team
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace Test {

template <class ExecSpace>
struct ThreadScratch {
  using policy_t = Kokkos::TeamPolicy<ExecSpace>;
  using team_t   = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using data_t   = Kokkos::View<size_t **, ExecSpace>;

  using scratch_t =
      Kokkos::View<size_t *, typename ExecSpace::scratch_memory_space_l0,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  int sX, sY;
  data_t v;

  static constexpr int scratch_level = 1;
  KOKKOS_FUNCTION
  void operator()(const team_t &team) const {
    // Allocate and use scratch pad memory
    scratch_t v_S =
        scratch_t(team.template thread_scratch<scratch_level>(), sY);
    sycl::local_ptr<size_t> v_S_ptr = v_S.data();
    Kokkos::printf("%p\n", v_S_ptr.get());
    int n = team.league_rank();

    for (int i = 0; i < sY; ++i) v_S[i] = 0;
  }

  void run(const int pN, const int sX_, const int sY_) {
    sX = sX_;
    sY = sY_;

    int scratchSize = scratch_t::shmem_size(sY);
    // So this works with deprecated code enabled:
    policy_t policy =
        policy_t(pN, 1, 1)
            .set_scratch_size(scratch_level, Kokkos::PerThread(scratchSize));

    int max_team_size = policy.team_size_max(*this, Kokkos::ParallelForTag());
    v                 = data_t("Matrix", pN, max_team_size);

    Kokkos::parallel_for(
        "Test12a_ThreadScratch",
        policy_t(pN, max_team_size)
            .set_scratch_size(scratch_level, Kokkos::PerThread(scratchSize)),
        *this);
  }
};

TEST(TEST_CATEGORY, IncrTest_12a_ThreadScratch) {
  ThreadScratch<TEST_EXECSPACE> test;
  test.run(1, 1, 1);
}

}  // namespace Test
