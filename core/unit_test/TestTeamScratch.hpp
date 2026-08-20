// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_TEAM_SCRATCH_HPP
#define KOKKOS_TEST_TEAM_SCRATCH_HPP
#include <type_traits>

#include <TestTeam.hpp>

namespace Test {

namespace {

template <class ExecSpace, int Level>
struct TestCompileTimeScratchLevelOverloadsFunctor {
  using member_type    = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  using flag_view_type = Kokkos::View<int, ExecSpace>;

  flag_view_type m_flag;

  KOKKOS_FUNCTION void operator()(const member_type &team) const {
    constexpr int allocation_size = 16;
    constexpr int alignment       = 8;

    const auto &scratch = team.team_shmem();

    auto *tag = static_cast<int *>(scratch.get_shmem(
        allocation_size, std::integral_constant<int, Level>{}));
    auto *explicit_level =
        static_cast<int *>(scratch.template get_shmem<Level>(allocation_size));
    auto *aligned_tag      = static_cast<int *>(scratch.get_shmem_aligned(
        allocation_size, alignment, std::integral_constant<int, Level>{}));
    auto *aligned_explicit = static_cast<int *>(
        scratch.template get_shmem_aligned<Level>(allocation_size, alignment));

    if ((reinterpret_cast<uintptr_t>(aligned_tag) % alignment != 0) ||
        (reinterpret_cast<uintptr_t>(aligned_explicit) % alignment != 0)) {
      m_flag() = 1;
      return;
    }

    // Unique values detect overlapping allocations from the four overloads.
    *tag              = 1;
    *explicit_level   = 2;
    *aligned_tag      = 3;
    *aligned_explicit = 4;

    if (*tag + *explicit_level + *aligned_tag + *aligned_explicit != 10) {
      m_flag() = 2;
    }
  }
};

template <class ExecSpace>
struct TestCompileTimeScratchLevelOverloads {
  TestCompileTimeScratchLevelOverloads() {
    // Test levels separately because host backends may alias their backing
    // storage.
    run<0>();
    run<1>();
  }

  template <int Level>
  void run() {
    constexpr int scratch_size = 64;

    Kokkos::View<int, ExecSpace> flag("flag");
    Kokkos::TeamPolicy<ExecSpace> policy(1, 1);
    policy.set_scratch_size(Level, Kokkos::PerTeam(scratch_size));

    Kokkos::parallel_for(
        "compile_time_scratch_level_overloads", policy,
        TestCompileTimeScratchLevelOverloadsFunctor<ExecSpace, Level>{flag});
    Kokkos::fence();

    int host_flag = 0;
    Kokkos::deep_copy(host_flag, flag);
    ASSERT_EQ(host_flag, 0) << "scratch level " << Level;
  }
};

}  // namespace

TEST(TEST_CATEGORY, team_shared_request) {
  TestSharedTeam<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> >();
  TestSharedTeam<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> >();
}

TEST(TEST_CATEGORY, team_scratch_request) {
  TestScratchTeam<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Static> >();
  TestScratchTeam<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic> >();
}

TEST(TEST_CATEGORY, team_lambda_shared_request) {
  TestLambdaSharedTeam<Kokkos::HostSpace, TEST_EXECSPACE,
                       Kokkos::Schedule<Kokkos::Static> >();
  TestLambdaSharedTeam<Kokkos::HostSpace, TEST_EXECSPACE,
                       Kokkos::Schedule<Kokkos::Dynamic> >();
}
TEST(TEST_CATEGORY, scratch_align) { TestScratchAlignment<TEST_EXECSPACE>(); }

TEST(TEST_CATEGORY, shmem_size) { TestShmemSize<TEST_EXECSPACE>(); }

TEST(TEST_CATEGORY, multi_level_scratch) {
  TestMultiLevelScratchTeam<TEST_EXECSPACE,
                            Kokkos::Schedule<Kokkos::Static> >();
  TestMultiLevelScratchTeam<TEST_EXECSPACE,
                            Kokkos::Schedule<Kokkos::Dynamic> >();
}

TEST(TEST_CATEGORY, scratch_compile_time_level_overloads) {
  TestCompileTimeScratchLevelOverloads<TEST_EXECSPACE>();
}

struct DummyTeamParallelForFunctor {
  KOKKOS_FUNCTION void operator()(
      Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type) const {}
};

TEST(TEST_CATEGORY, team_scratch_memory_index_parallel_for) {
  // Requesting per team scratch memory for a largish number of teams, resulted
  // in problems computing the correct scratch pointer due to missed
  // initialization of the maximum number of scratch pad indices in the Cuda
  // baackend.
  const int scratch_size = 4896;
  const int league_size  = 7535;

  Kokkos::TeamPolicy<TEST_EXECSPACE> policy(league_size, Kokkos::AUTO);
  policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size));
  Kokkos::parallel_for("kernel", policy, DummyTeamParallelForFunctor());
}

TEST(TEST_CATEGORY, scratch_size_query) {
  const int thread_scratch[] = {4, 32};
  const int team_scratch[]   = {64, 256};
  const int league_size      = 10;
  const int team_size        = 1;

  Kokkos::TeamPolicy<TEST_EXECSPACE> policy(league_size, team_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(team_scratch[0]),
                          Kokkos::PerThread(thread_scratch[0]));
  policy.set_scratch_size(1, Kokkos::PerTeam(team_scratch[1]),
                          Kokkos::PerThread(thread_scratch[1]));

  ASSERT_EQ(policy.team_scratch_size(0), team_scratch[0]);
  ASSERT_EQ(policy.team_scratch_size(1), team_scratch[1]);
  ASSERT_EQ(policy.thread_scratch_size(0), thread_scratch[0]);
  ASSERT_EQ(policy.thread_scratch_size(1), thread_scratch[1]);
}

}  // namespace Test
#endif
