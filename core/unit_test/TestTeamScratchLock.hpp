// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_TEAM_SCRATCH_LOCK_HPP
#define KOKKOS_TEST_TEAM_SCRATCH_LOCK_HPP

#include <algorithm>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Stress test for the lock that guards level-1 team scratch on GPU backends
// (cuda/hip/sycl_{get,release}_scratch_index). Only the public API is used, so
// the same test runs on every backend.
//
// The number of locks is bounded by the execution space concurrency, so with a
// league much larger than that, teams are forced to hand scratch slots over to
// each other. If the unlock does not have release semantics (or the lock does
// not have acquire semantics), delayed writes of the previous owner can land
// after the next owner has written, or two teams can use a slot concurrently.
// Both show up as a mismatch when the owner reads back what it just wrote.
// Scratch is never read before it has been written in the same team.

namespace {

template <class ExecSpace>
int run(const int rep, const int team_size) {
  using policy_t  = Kokkos::TeamPolicy<ExecSpace>;
  using member_t  = typename policy_t::member_type;
  using scratch_t = Kokkos::View<int*, typename ExecSpace::scratch_memory_space,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  ExecSpace exec;

  if (policy_t{exec, 1, 1}.team_size_max(KOKKOS_LAMBDA(const member_t&){},
                                         Kokkos::ParallelForTag{}) < team_size)
    return 0;
  const int num_words = 256;

  const int league_size = std::max<int>(
      200000, 16 * (static_cast<int>(exec.concurrency()) / team_size));

  policy_t policy(exec, league_size, team_size);
  policy.set_scratch_size(1, Kokkos::PerTeam(num_words * sizeof(int)));

  // Scratch memory is uninitialized when a team obtains it, so it must be
  // written before it is read. Every team fills its slot with a value unique
  // to the team, synchronizes, and reads the slot back. A previous owner whose
  // writes are not ordered before the lock release can overwrite the slot
  // afterwards, and two teams using the same slot concurrently clobber each
  // other; both show up as a mismatch.
  Kokkos::View<int, typename ExecSpace::memory_space> errors_d("errors");

  Kokkos::parallel_for(
      "team_scratch_lock", policy, KOKKOS_LAMBDA(const member_t& team) {
        scratch_t scratch(team.team_scratch(1), num_words);
        const int value = (0x40 + rep) * 1000003 + team.league_rank();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, num_words),
                             [&](const int w) { scratch(w) = value + w; });
        team.team_barrier();

        int bad = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, num_words),
            [&](const int w, int& e) { e += (scratch(w) != value + w); }, bad);
        team.team_barrier();

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          if (bad != 0) Kokkos::atomic_add(&errors_d(), bad);
        });
      });
  exec.fence();

  int errors = 0;
  Kokkos::deep_copy(errors, errors_d);
  return errors;
}

TEST(TEST_CATEGORY, team_scratch_lock_handoff) {
  // Team sizes for which the number of lock slots is not a multiple of the
  // number of GPU dies, so that consecutive owners of a slot run on different
  // dies (no shared cache) and a missing release/acquire becomes visible.
  // Repeat to give a racy implementation more chances to fail.
  for (int team_size : {3, 5, 7, 11})
    for (int rep = 0; rep < 3; ++rep)
      EXPECT_EQ(0, run<TEST_EXECSPACE>(rep, team_size))
          << "team_size " << team_size << " repetition " << rep;
}

}  // namespace

#endif
