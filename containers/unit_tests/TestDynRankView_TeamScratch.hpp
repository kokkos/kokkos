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

#include <Kokkos_DynRankView.hpp>

namespace {

void test_dyn_rank_view_team_scratch() {
  using execution_space = TEST_EXECSPACE;
  using memory_space    = execution_space::scratch_memory_space;
  using drv_type        = Kokkos::DynRankView<int, memory_space>;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using team_type       = policy_type::member_type;

  int N0 = 10, N1 = 4, N2 = 3;
  size_t shmem_size = drv_type::shmem_size(N0, N1, N2);
  ASSERT_GE(shmem_size, N0 * N1 * N2 * sizeof(int));

  Kokkos::View<unsigned, execution_space, Kokkos::MemoryTraits<Kokkos::Atomic>>
      errors("errors");
  auto policy = policy_type(1, Kokkos::AUTO)
                    .set_scratch_size(0, Kokkos::PerTeam(shmem_size));
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const team_type& team) {
        drv_type scr(team.team_scratch(0), N0, N1, N2);
        // Control that the code ran at all
        if (scr.rank() != 3) errors() |= 1u;
        if (scr.extent_int(0) != N0) errors() |= 2u;
        if (scr.extent_int(1) != N1) errors() |= 4u;
        if (scr.extent_int(2) != N2) errors() |= 8u;
        Kokkos::parallel_for(
            Kokkos::TeamThreadMDRange(team, N0, N1, N2),
            [=](int i, int j, int k) { scr(i, j, k) = i * 100 + j * 10 + k; });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadMDRange(team, N0, N1, N2),
                             [=](int i, int j, int k) {
                               if (scr(i, j, k) != i * 100 + j * 10 + k)
                                 errors() |= 16u;
                             });
        errors() |= 256u;
      });
  unsigned h_errors = 0;
  Kokkos::deep_copy(h_errors, errors);

  ASSERT_EQ((h_errors & 1u), 0u) << "Rank mismatch";
  ASSERT_EQ((h_errors & 2u), 0u) << "extent 0 mismatch";
  ASSERT_EQ((h_errors & 4u), 0u) << "extent 1 mismatch";
  ASSERT_EQ((h_errors & 8u), 0u) << "extent 2 mismatch";
  ASSERT_EQ((h_errors & 16u), 0u) << "data access incorrect";
  ASSERT_EQ(h_errors, 256u);
}

TEST(TEST_CATEGORY, dyn_rank_view_team_scratch) {
  test_dyn_rank_view_team_scratch();
}

}  // namespace
