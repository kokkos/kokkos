// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_LOCKPOLICY_HPP
#define KOKKOS_LOCKPOLICY_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

#include <impl/Kokkos_ClockTic.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace Kokkos {
namespace Experimental {

// ==============================================================================
// Internal utilities (hardware thread id, seeding, and backoff helpers)
// ==============================================================================
namespace Impl {

// Default cap on the busy-wait delay (in clock_tic() units, or instruction
// cost) used by backoff policies.
inline constexpr uint32_t default_backoff_max_delay = 512;

// A hardware thread identifier (device / host).
// This is only used to seed per-thread pseudo-random backoff jitter, never
// as a lock key or as a stable/unique thread identity.
KOKKOS_INLINE_FUNCTION uint32_t get_hardware_thread_id() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // Unique global lane/thread id on GPU.
  return static_cast<uint32_t>(
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z) +
      (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) *
          (blockDim.x * blockDim.y * blockDim.z));
#else
  // On host: combine the calling thread's stack address with the clock.
  int dummy = 0;
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&dummy) ^
                               Kokkos::Impl::clock_tic());
#endif
}

// Generates a pseudo-random seed from the hardware thread id and clock_tic().
// The tic is used to have a different initial state between 2 calls.
KOKKOS_INLINE_FUNCTION uint32_t generate_thread_seed() {
  uint32_t tid = get_hardware_thread_id();
  auto tic     = static_cast<uint64_t>(Kokkos::Impl::clock_tic());
  // Combine the tid and tic together as original seed
  uint32_t seed = tid ^ static_cast<uint32_t>(tic ^ (tic >> 32));

  // Thomas Wang's hash function to spread entropy.
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  return seed ^ (seed >> 15);
}

// Draws a delay in [0, max_delay] from `seed`, then advances `seed` so the next
// draw differs. `max_delay` is assumed well below UINT32_MAX (true for any sane
// backoff cap).
KOKKOS_INLINE_FUNCTION uint32_t next_random_delay(uint32_t& seed,
                                                  uint32_t max_delay) {
  uint32_t delay = seed % (max_delay + 1);
  seed           = seed * 1103515245 + 12345;
  return delay;
}

// Single compare-and-swap (CAS) attempt to acquire the lock.
template <typename LockType>
KOKKOS_INLINE_FUNCTION bool try_lock_cas(LockType* lock) {
  return Kokkos::atomic_compare_exchange(lock, LockType(0), LockType(1)) ==
         LockType(0);
}

// Single test-then-test-and-set (TTAS) attempt to acquire the lock.
template <typename LockType>
KOKKOS_INLINE_FUNCTION bool try_lock_ttas(LockType* lock) {
  return Kokkos::atomic_load(lock) == LockType(0) &&
         Kokkos::atomic_compare_exchange(lock, LockType(0), LockType(1)) ==
             LockType(0);
}

// TryLock strategies: the CAS/TTAS axis of variation, factored out so
// BackoffLockPolicy (below) can be built once and combined with either.
struct CasTryLock {
  template <typename LockType>
  KOKKOS_INLINE_FUNCTION static bool try_lock(LockType* lock) {
    return try_lock_cas(lock);
  }
};

struct TtasTryLock {
  template <typename LockType>
  KOKKOS_INLINE_FUNCTION static bool try_lock(LockType* lock) {
    return try_lock_ttas(lock);
  }
};

// Backoff strategies: the "how do we wait between attempts" axis of variation.
// Each type is constructed fresh at the start of every `acquire()` call and
// exposes `on_failed_attempt()`, called once per failed try_lock.

// Exponential backoff timed by counting load_fence() iterations: cheap, but the
// actual wait scales with instruction cost, not wall time.
class TickExponentialBackoff {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit TickExponentialBackoff(uint32_t max_delay)
      : m_max_delay(max_delay), m_delay(1) {}

  KOKKOS_INLINE_FUNCTION void on_failed_attempt() {
    for (uint32_t tick = 0; tick < m_delay; ++tick) {
      // The fence should keep the compiler from proving this loop has no
      // observable effect and hoisting or collapsing it away, and forces a
      // fresh read each iteration.
      Kokkos::load_fence();
    }
    if (m_delay < m_max_delay) m_delay *= 2;
  }

 private:
  uint32_t m_max_delay;
  uint32_t m_delay;
};

// Exponential backoff timed against Kokkos' abstract cycle counter
// clock_tic(). This is closer to actual elapsed time regardless of how
// expensive load_fence() happens to be on a given backend.

// Spinning on the clock instead of retrying the atomic keeps traffic off the
// lock's cache line while still not requiring an OS-level sleep, which is
// not available on device.
class ClockExponentialBackoff {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit ClockExponentialBackoff(uint32_t max_delay)
      : m_max_delay(max_delay), m_attempt(0) {}

  // `m_max_delay` bounds the actual wait time (in clock_tic() units), so
  // callers can tune how long a single backoff step may last. `max_shift` below
  // is a separate, fixed safety cap on the shift amount itself, only there to
  // avoid undefined behavior / overflow if `m_attempt` grows very large.
  KOKKOS_INLINE_FUNCTION void on_failed_attempt() {
    auto start = Kokkos::Impl::clock_tic();

    constexpr int max_shift = 17;
    int shift               = (m_attempt < max_shift) ? m_attempt : max_shift;
    uint32_t raw_delay      = (static_cast<uint32_t>(1) << shift);
    auto delay              = static_cast<decltype(start)>(
        raw_delay < m_max_delay ? raw_delay
                                             : static_cast<uint64_t>(m_max_delay));

    while (static_cast<decltype(start)>(Kokkos::Impl::clock_tic()) - start <
           delay) {
      // Active, non-blocking wait on the cycle counter.
      Kokkos::load_fence();
    }
    ++m_attempt;
  }

 private:
  uint32_t m_max_delay;
  int m_attempt;
};

// Random backoff timed by counting load_fence() iterations.
class TickRandomBackoff {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit TickRandomBackoff(uint32_t max_delay)
      : m_max_delay(max_delay), m_seed(generate_thread_seed()) {}

  KOKKOS_INLINE_FUNCTION void on_failed_attempt() {
    uint32_t delay = next_random_delay(m_seed, m_max_delay);
    for (uint32_t tick = 0; tick < delay; ++tick) {
      Kokkos::load_fence();
    }
  }

 private:
  uint32_t m_max_delay;
  uint32_t m_seed;
};

// Random backoff timed against Kokkos' abstract cycle counter clock_tic().
class ClockRandomBackoff {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit ClockRandomBackoff(uint32_t max_delay)
      : m_max_delay(max_delay), m_seed(generate_thread_seed()) {}

  KOKKOS_INLINE_FUNCTION void on_failed_attempt() {
    uint32_t delay = next_random_delay(m_seed, m_max_delay);
    auto start     = Kokkos::Impl::clock_tic();
    while (static_cast<decltype(start)>(Kokkos::Impl::clock_tic()) - start <
           delay) {
      Kokkos::load_fence();
    }
  }

 private:
  uint32_t m_max_delay;
  uint32_t m_seed;
};

// Combines a TryLock strategy (CasTryLock / TtasTryLock) with a Backoff
// type into a full LockPolicy. LockPolicy::ExponentialBackoff,
// ::RandomBackoffTTAS, etc. (below) are aliases of this template.
template <typename TryLock, typename Backoff>
class BackoffLockPolicy {
 public:
  static constexpr uint32_t default_max_delay = default_backoff_max_delay;

  KOKKOS_INLINE_FUNCTION
  constexpr BackoffLockPolicy() : m_max_delay(default_max_delay) {}

  // Caps how long a single backoff step may wait (ticks or clock_tic()
  // units, depending on Backoff).
  KOKKOS_INLINE_FUNCTION
  explicit constexpr BackoffLockPolicy(uint32_t max_delay)
      : m_max_delay(max_delay) {}

  template <typename LockType>
  KOKKOS_INLINE_FUNCTION void acquire(LockType* lock) const {
    Backoff backoff(m_max_delay);
    while (!TryLock::try_lock(lock)) {
      backoff.on_failed_attempt();
    }
  }

  template <typename LockType>
  KOKKOS_INLINE_FUNCTION void release(LockType* lock) const {
    Kokkos::memory_fence();
    Kokkos::atomic_store(lock, LockType(0));
  }

 private:
  uint32_t m_max_delay;
};

// RAII guard. Using a guard instead of a manual acquire/action/release
// sequence to ensure that the lock is still released if `action` throws on the
// host side (device code never throws exeception, but a host lambda passed to
// the same generic API might).
template <typename LockType, typename LockPolicy>
class LockGuard {
 public:
  KOKKOS_INLINE_FUNCTION
  LockGuard(LockType* lock, LockPolicy policy)
      : m_lock(lock), m_policy(policy) {
    m_policy.acquire(m_lock);
  }

  KOKKOS_INLINE_FUNCTION
  ~LockGuard() { m_policy.release(m_lock); }

  LockGuard(const LockGuard&)            = delete;
  LockGuard& operator=(const LockGuard&) = delete;
  LockGuard(LockGuard&&)                 = delete;
  LockGuard& operator=(LockGuard&&)      = delete;

 private:
  LockType* m_lock;
  LockPolicy m_policy;
};

// ExecutionSpace traits used for backend dispatch in atomic_locked_action
template <typename ExecutionSpace>
inline constexpr bool is_serial_execution_space_v = false;

#if defined(KOKKOS_ENABLE_SERIAL)
template <>
inline constexpr bool is_serial_execution_space_v<Kokkos::Serial> = true;
#endif

template <typename ExecutionSpace>
inline constexpr bool is_hip_execution_space_v = false;

#if defined(KOKKOS_ENABLE_HIP)
template <>
inline constexpr bool is_hip_execution_space_v<Kokkos::HIP> = true;
#endif

template <typename ExecutionSpace>
inline constexpr bool is_cuda_execution_space_v = false;

#if defined(KOKKOS_ENABLE_CUDA)
template <>
inline constexpr bool is_cuda_execution_space_v<Kokkos::Cuda> = true;
#endif

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
// Runs `action` atomically under `lock`, safe for SIMT execution groups that
// do not guarantee independent forward progress between lanes: AMD
// wavefronts (always, on every ROCm/architecture combination we know of),
// and NVIDIA warps on architectures older than Volta (compute capability
// < 7.0), which predate Independent Thread Scheduling.
//
// This deliberately avoids warp-vote/shuffle intrinsics (__ballot, __shfl,
// __activemask, and friends) for the lane-election logic itself. Their use
// under divergent control flow (like here) was affected by at least a confirmed
// HIP compiler bug that produced wrong results (see ROCm/hip#952 and the
// follow-up ROCm/hip#2474), reported present as late as 2022 and fixed in 2023.
// The `_sync` variants (explicit participation mask) became available, and
// enabled by default, starting with ROCm 7.0. Even though current ROCm is
// expected to have this fixed, the plain uniform-loop approach below avoids
// depending on that fix being present for whichever ROCm version and hardware
// this ends up compiled against, and sidesteps the question entirely.
//
// Instead, every lane in the group runs the "same" uniform, convergent
// loop over lane indices. The loop structure never depends on any
// other lane's state. Within that loop, only the lane whose own index
// matches the current iteration ever touches `lock`, every other lane's
// iteration is a no-op. This means at most one lane per group is ever
// inside the acquire/action/release sequence at a time, which is what
// actually removes the lockstep hazard. Because only one lane is ever
// spinning at a time, it's safe to reuse the same LockPolicy types used by
// the generic path for the elected lane's own acquire/release.
//
// Cost: this serializes the whole group through this call, one lane at
// a time, even when the individual locks are all different and
// uncontended. It trades away all intra-group parallelism for this
// operation in exchange for correctness on this hardware. (A better solution
// could be considered, but it is not trivial.)
//
// Open question for a later pass: given the failure was a compiler bug and
// not a design limitation, would it be worth maintaining a second, faster
// implementation on top of __ballot/__shfl for ROCm versions known to have
// the fix, selected at compile time via HIP_VERSION (or
// HIP_VERSION_MAJOR/HIP_VERSION_MINOR), falling back to this uniform loop
// to recover intra-group parallelism for uncontended locks?
//
// NOTE: This does NOT remove the more general (and universal, Volta+ CUDA
// included) concern that a lock could be held by a thread in a different
// group that isn't currently scheduled. This also apply to CPU if a thread is
// never rescheduled by the kernel, that's a launch/occupancy concern for any
// hardware spinlock, not something fixable at this layer.
template <typename LockType, typename LockPolicy, typename Function>
KOKKOS_INLINE_FUNCTION decltype(auto) lane_serialized_locked_action(
    LockType* lock, LockPolicy policy, Function&& action) {
  KOKKOS_IF_ON_HOST((Impl::LockGuard<LockType, LockPolicy> guard(lock, policy);
                     return action();))

  KOKKOS_IF_ON_DEVICE((
      unsigned int flat_local_id =
          threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
      unsigned int my_lane =
          flat_local_id % static_cast<unsigned int>(warpSize);

      using ReturnType = decltype(action());

      // Handles void and non-void return type.
      if constexpr (std::is_void_v<ReturnType>) {
        for (unsigned int elected_lane = 0;
             elected_lane < static_cast<unsigned int>(warpSize);
             ++elected_lane) {
          if (my_lane == elected_lane) {
            Impl::LockGuard<LockType, LockPolicy> guard(lock, policy);
            action();
          }
        }
      } else {
        static_assert(
            std::is_default_constructible_v<ReturnType>,
            "atomic_locked_action's lane-serialized path requires the "
            "action's return type to be default-constructible (or void).");
        ReturnType result{};
        for (unsigned int elected_lane = 0;
             elected_lane < static_cast<unsigned int>(warpSize);
             ++elected_lane) {
          if (my_lane == elected_lane) {
            Impl::LockGuard<LockType, LockPolicy> guard(lock, policy);
            result = action();
          }
        }
        // The return is performed uniformly by each active lane outside the
        // loop to avoid divergence.
        return result;
      }))
}
#endif  // KOKKOS_ENABLE_HIP || KOKKOS_ENABLE_CUDA

#if defined(KOKKOS_ENABLE_CUDA)
// CUDA-specific dispatch: routes to the lane-serialized path above only on
// pre-Volta architectures (compute capability < 7.0).
template <typename LockType, typename LockPolicy, typename Function>
KOKKOS_INLINE_FUNCTION decltype(auto) cuda_dispatch_locked_action(
    LockType* lock, LockPolicy policy, Function&& action) {
  KOKKOS_IF_ON_HOST((Impl::LockGuard<LockType, LockPolicy> guard(lock, policy);
                     return action();))

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  KOKKOS_IF_ON_DEVICE((
      return Impl::lane_serialized_locked_action(
          lock, policy, std::forward<Function>(action));
  ))
#else
  KOKKOS_IF_ON_DEVICE((
      Impl::LockGuard<LockType, LockPolicy> guard(lock, policy);
      return action();
  ))
#endif
}
#endif  // KOKKOS_ENABLE_CUDA

}  // namespace Impl

// ==============================================================================
// Kokkos::Experimental::LockPolicy
// ==============================================================================
namespace LockPolicy {
// A LockPolicy type must be default-constructible, copyable, and expose two
// member functions, callable from both host and device:
//
//   template <typename LockType> void acquire(LockType* lock) const;
//   template <typename LockType> void release(LockType* lock) const;
//
// `lock` points at a zero-initialized location (0 = free, 1 = held).
//
// NOTE: taken on its own, those are naive algorithms with no special handling
// for GPUs that do not guarantee independent forward progress within a
// warp/wavefront (e.g. AMD HIP, and NVIDIA pre-Volta): two lanes of the same
// group contending for the same lock can deadlock if one holds the lock
// while another spins here, because the hardware may not reschedule the
// holder until the waiter's SIMD step completes. `atomic_locked_action`
// (below) handles this correctly for `ExecutionSpace = Kokkos::HIP` and for
// pre-Volta `Kokkos::Cuda` via lane-serialized dispatch, see
// `Impl::lane_serialized_locked_action` (above). Using these LockPolicy
// types directly on such hardware (or any other lacking forward progress)
// outside a carefully crafted dispatch remains unsafe.

// ==============================================================================
// SPINLOCK POLICIES
// ==============================================================================

// Pure spinlock: retries the CAS immediately. Kept deliberately minimal.
struct PureSpinlock {
  template <typename LockType>
  KOKKOS_INLINE_FUNCTION void acquire(LockType* lock) const {
    while (!Impl::try_lock_cas(lock)) {
      // The fence should prevent the compiler from optimizing this loop away
      // and forces a fresh attempt each iteration.
      Kokkos::load_fence();
    }
  }

  template <typename LockType>
  KOKKOS_INLINE_FUNCTION void release(LockType* lock) const {
    Kokkos::memory_fence();  // Ensure writes made in the critical section
                             // are visible before the lock is cleared.
    Kokkos::atomic_store(lock, LockType(0));
  }
};

// Spinlock with test-then-test-and-set: polls with a plain load
// before attempting the CAS, so contending threads are not all
// hammering the atomic unit / cache line with a CAS while the lock
// is held by someone else.
struct SpinlockTTAS {
  template <typename LockType>
  KOKKOS_INLINE_FUNCTION void acquire(LockType* lock) const {
    while (!Impl::try_lock_ttas(lock)) {
      Kokkos::load_fence();
    }
  }

  template <typename LockType>
  KOKKOS_INLINE_FUNCTION void release(LockType* lock) const {
    Kokkos::memory_fence();
    Kokkos::atomic_store(lock, LockType(0));
  }
};

// ==============================================================================
// EXPONENTIAL BACKOFF POLICIES
// ==============================================================================
//
// All four are Impl::BackoffLockPolicy instantiations (CAS/TTAS combined
// with a tick- or clock-timed exponential backoff state).

using ExponentialBackoff =
    Impl::BackoffLockPolicy<Impl::CasTryLock, Impl::TickExponentialBackoff>;
using ExponentialBackoffTTAS =
    Impl::BackoffLockPolicy<Impl::TtasTryLock, Impl::TickExponentialBackoff>;
using ClockExponentialBackoff =
    Impl::BackoffLockPolicy<Impl::CasTryLock, Impl::ClockExponentialBackoff>;
using ClockExponentialBackoffTTAS =
    Impl::BackoffLockPolicy<Impl::TtasTryLock, Impl::ClockExponentialBackoff>;

// ==============================================================================
// RANDOM BACKOFF POLICIES
// ==============================================================================
//
// Same idea as the exponential family above: CAS/TTAS combined with a
// tick- or clock-timed random backoff state.

using RandomBackoff =
    Impl::BackoffLockPolicy<Impl::CasTryLock, Impl::TickRandomBackoff>;
using RandomBackoffTTAS =
    Impl::BackoffLockPolicy<Impl::TtasTryLock, Impl::TickRandomBackoff>;
using ClockRandomBackoff =
    Impl::BackoffLockPolicy<Impl::CasTryLock, Impl::ClockRandomBackoff>;
using ClockRandomBackoffTTAS =
    Impl::BackoffLockPolicy<Impl::TtasTryLock, Impl::ClockRandomBackoff>;

}  // namespace LockPolicy

// ==============================================================================
// atomic_locked_action: runs a user action atomically while holding a lock,
// dispatching on ExecutionSpace and using the given LockPolicy to decide how
// to wait for the lock.
//
// Dispatch:
//   - ExecutionSpace = Kokkos::Serial: no-op lock -- a single thread can
//     never contend with itself, so `action` just runs directly.
//   - ExecutionSpace = Kokkos::HIP: lane-serialized dispatch (see
//     Impl::lane_serialized_locked_action) to avoid the lockstep/forward-
//     progress hazard.
//   - ExecutionSpace = Kokkos::Cuda: lane-serialized dispatch too, but only
//     on pre-Volta architectures (compute capability < 7.0). Volta and later
//     have Independent Thread Scheduling and use the generic path below
//     instead.
//   - Everything else (OpenMP, Threads, Volta+ Cuda, SYCL, HPX, ...): the
//     generic LockPolicy-based spin loop via Impl::LockGuard.
//
// Four overloads are provided, differing in how much you want to specify
// explicitly (LockType and Function are always deduced from `lock` and
// `action`):
//
//   atomic_locked_action<ES>(lock, my_policy_instance, action);       // (1)
//   atomic_locked_action<ES, LockPolicy::SpinlockTTAS>(lock, action); // (2)
//   atomic_locked_action<ES>(lock, action);                           // (3)
//   atomic_locked_action(lock, action);                               // (4)
//
// (3) uses LockPolicy::ExponentialBackoff by default;
// (4) additionally defaults ExecutionSpace to Kokkos::DefaultExecutionSpace.
// ==============================================================================

/**
 * @brief Acquires `lock` using `policy`, runs `action` atomically, then
 * releases `lock`. Dispatch using `ExecutionSpace`.
 *
 * @tparam ExecutionSpace The Kokkos execution space this call runs under.
 * @tparam LockType   Underlying type of the lock word (e.g. int32_t,
 *                     uint32_t). Must be an integral type supported by
 *                     Kokkos atomics.
 * @tparam LockPolicy Policy instance type (e.g. LockPolicy::PureSpinlock).
 * @tparam Function   User function/lambda type (e.g. KOKKOS_LAMBDA). May
 *                     return void or a value; a returned value is forwarded
 *                     back to the caller.
 */
template <typename ExecutionSpace, typename LockType, typename LockPolicy,
          typename Function>
KOKKOS_INLINE_FUNCTION decltype(auto) atomic_locked_action(LockType* lock,
                                                           LockPolicy policy,
                                                           Function&& action) {
  static_assert(std::is_integral_v<LockType>,
                "LockType must be an integral type supported by Kokkos "
                "atomics.");

  if constexpr (Impl::is_serial_execution_space_v<ExecutionSpace>) {
    // Serial: only one thread ever exists, so nothing else can be
    // contending for this lock. Skip the atomic dance entirely.
    return action();
  }
#if defined(KOKKOS_ENABLE_HIP)
  else if constexpr (Impl::is_hip_execution_space_v<ExecutionSpace>) {
    return Impl::lane_serialized_locked_action(lock, policy,
                                               std::forward<Function>(action));
  }
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  else if constexpr (Impl::is_cuda_execution_space_v<ExecutionSpace>) {
    return Impl::cuda_dispatch_locked_action(lock, policy,
                                             std::forward<Function>(action));
  }
#endif
  else {
    Impl::LockGuard<LockType, LockPolicy> guard(lock, policy);
    return action();
  }
}

// Overload (2): explicit ExecutionSpace and explicit LockPolicy type,
// default-constructed instead of passed as an instance. Delegates to the
// overload above so the dispatch logic is defined in exactly one place.
template <typename ExecutionSpace, typename LockPolicy, typename LockType,
          typename Function>
KOKKOS_INLINE_FUNCTION decltype(auto) atomic_locked_action(LockType* lock,
                                                           Function&& action) {
  return atomic_locked_action<ExecutionSpace>(lock, LockPolicy{},
                                              std::forward<Function>(action));
}

// Overload (3): explicit ExecutionSpace, default LockPolicy
// (ExponentialBackoff).
template <
    typename ExecutionSpace, typename LockType, typename Function,
    typename LockPolicy = Kokkos::Experimental::LockPolicy::ExponentialBackoff>
KOKKOS_INLINE_FUNCTION decltype(auto) atomic_locked_action(LockType* lock,
                                                           Function&& action) {
  return atomic_locked_action<ExecutionSpace>(lock, LockPolicy{},
                                              std::forward<Function>(action));
}

// Overload (4): fully implicit, ExecutionSpace defaults to
// Kokkos::DefaultExecutionSpace, LockPolicy defaults to ExponentialBackoff.
template <
    typename LockType, typename Function,
    typename ExecutionSpace = Kokkos::DefaultExecutionSpace,
    typename LockPolicy = Kokkos::Experimental::LockPolicy::ExponentialBackoff>
KOKKOS_INLINE_FUNCTION decltype(auto) atomic_locked_action(LockType* lock,
                                                           Function&& action) {
  return atomic_locked_action<ExecutionSpace>(lock, LockPolicy{},
                                              std::forward<Function>(action));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_LOCKPOLICY_HPP
