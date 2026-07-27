// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOS_TEST_LOCKPOLICY_HPP
#define KOKKOS_TEST_LOCKPOLICY_HPP

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <Kokkos_LockPolicy.hpp>

namespace Test {

// Generic helper function to test lock contention for a given policy and
// execution space
template <typename ExecutionSpace, typename LockPolicy>
void run_lock_contention_test(const int num_increments) {
  // The lock and counter must be accessible on the target ExecutionSpace.
  // A scalar View is initialized to zero by default, which matches
  // the free state expected by try_lock and LockGuard (0 = free).
  Kokkos::View<int32_t, ExecutionSpace> counter("counter");
  Kokkos::View<int32_t, ExecutionSpace> lock("lock");

  Kokkos::parallel_for(
      "TestLockContention",
      Kokkos::RangePolicy<ExecutionSpace>(0, num_increments),
      KOKKOS_LAMBDA(const int) {
        LockPolicy policy{};
        // Pass the lock address and execute the atomic increment within the
        // critical section.
        Kokkos::Experimental::atomic_locked_action<ExecutionSpace>(
            &lock(), policy, [=]() { counter()++; });
      });

  Kokkos::fence("Fence after lock contention test");

  // Copy result back to host for GTest verification
  auto h_counter =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counter);

  EXPECT_EQ(h_counter(), num_increments);
}

// GTest test definitions for each LockPolicy

TEST(TEST_CATEGORY, lock_policy_pure_spinlock) {
  // Test using PureSpinlock policy
  run_lock_contention_test<TEST_EXECSPACE,
                           Kokkos::Experimental::LockPolicy::PureSpinlock>(
      10000);
}

TEST(TEST_CATEGORY, lock_policy_spinlock_ttas) {
  // Test using SpinlockTTAS policy
  run_lock_contention_test<TEST_EXECSPACE,
                           Kokkos::Experimental::LockPolicy::SpinlockTTAS>(
      10000);
}

TEST(TEST_CATEGORY, lock_policy_exponential_backoff) {
  // Test using ExponentialBackoff policy (default for overloads 3 and 4)
  run_lock_contention_test<
      TEST_EXECSPACE, Kokkos::Experimental::LockPolicy::ExponentialBackoff>(
      10000);
}

TEST(TEST_CATEGORY, lock_policy_exponential_backoff_ttas) {
  // Test using ExponentialBackoffTTAS policy
  run_lock_contention_test<
      TEST_EXECSPACE, Kokkos::Experimental::LockPolicy::ExponentialBackoffTTAS>(
      10000);
}

TEST(TEST_CATEGORY, lock_policy_clock_exponential_backoff) {
  // Test using ClockExponentialBackoff policy
  run_lock_contention_test<
      TEST_EXECSPACE,
      Kokkos::Experimental::LockPolicy::ClockExponentialBackoff>(10000);
}

TEST(TEST_CATEGORY, lock_policy_clock_exponential_backoff_ttas) {
  // Test using ClockExponentialBackoffTTAS policy
  run_lock_contention_test<
      TEST_EXECSPACE,
      Kokkos::Experimental::LockPolicy::ClockExponentialBackoffTTAS>(10000);
}

TEST(TEST_CATEGORY, lock_policy_random_backoff) {
  // Test using RandomBackoff policy
  run_lock_contention_test<TEST_EXECSPACE,
                           Kokkos::Experimental::LockPolicy::RandomBackoff>(
      10000);
}

TEST(TEST_CATEGORY, lock_policy_random_backoff_ttas) {
  // Test using RandomBackoffTTAS policy
  run_lock_contention_test<TEST_EXECSPACE,
                           Kokkos::Experimental::LockPolicy::RandomBackoffTTAS>(
      10000);
}

TEST(TEST_CATEGORY, lock_policy_clock_random_backoff) {
  // Test using ClockRandomBackoff policy
  run_lock_contention_test<
      TEST_EXECSPACE, Kokkos::Experimental::LockPolicy::ClockRandomBackoff>(
      10000);
}

TEST(TEST_CATEGORY, lock_policy_clock_random_backoff_ttas) {
  // Test using ClockRandomBackoffTTAS policy
  run_lock_contention_test<
      TEST_EXECSPACE, Kokkos::Experimental::LockPolicy::ClockRandomBackoffTTAS>(
      10000);
}

// Test overload (4) which defaults ExecutionSpace to DefaultExecutionSpace
// and LockPolicy to ExponentialBackoff.
template <typename ExecutionSpace>
void run_lock_contention_test_default(const int num_increments) {
  Kokkos::View<int32_t, ExecutionSpace> counter("counter");
  Kokkos::View<int32_t, ExecutionSpace> lock("lock");

  Kokkos::parallel_for(
      "TestLockContentionImplicit",
      Kokkos::RangePolicy<ExecutionSpace>(0, num_increments),
      KOKKOS_LAMBDA(const int) {
        // Calls overload (4): fully implicit ExecutionSpace and LockPolicy.
        Kokkos::Experimental::atomic_locked_action(&lock(),
                                                   [=]() { counter()++; });
      });

  Kokkos::fence("Fence after implicit lock test");
  auto h_counter =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counter);
  EXPECT_EQ(h_counter(), num_increments);
}

TEST(TEST_CATEGORY, lock_policy_implicit_overloads) {
  run_lock_contention_test_default<TEST_EXECSPACE>(10000);
}

}  // namespace Test

#endif  // KOKKOS_TEST_LOCKPOLICY_HPP
