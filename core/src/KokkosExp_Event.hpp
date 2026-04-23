// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file KokkosExp_Event.hpp
/// \brief Experimental event API for fine-grained stream dependencies.
///
/// Events capture a point in an execution space's asynchronous timeline.
/// They enable cross-stream dependencies without a full fence, and
/// selective host synchronisation.
///
/// API:
///   - event.add_dependency_to(exec_space) — member: GPU-side dependency
///   (non-blocking on host)
///   - space_depends_on(exec_space, event) — free function: same, Kokkos-style
///   spelling
///   - event.wait()                        — host-side blocking synchronisation
///   - event.is_complete()                 — non-blocking query
///
/// Currently only the CUDA backend provides a native implementation.
/// For other backends the fallback records a fence on record() and
/// add_dependency_to / wait / is_complete are no-ops or trivially satisfied.

#ifndef KOKKOS_EXPERIMENTAL_EVENT_HPP
#define KOKKOS_EXPERIMENTAL_EVENT_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <Cuda/Kokkos_Cuda_Event.hpp>
#endif

namespace Kokkos {
namespace Experimental {

//============================================================================
// Backend-agnostic Event — fallback for non-native-event backends
//============================================================================

///  Portable fallback event for backends without native event support.
///
/// On record(), a fence is issued so that subsequent add_dependency_to() /
/// space_depends_on() and wait() are trivially satisfied.  This preserves
/// correctness at the cost of synchronisation -- the same trade-off existing
/// Kokkos code already makes.
///
/// Backends that provide a native implementation (e.g. CUDA) specialize
/// this template below.
template <class ExecutionSpace = DefaultExecutionSpace>
class Event {
 public:
  Event() = default;

  void record(const ExecutionSpace& exec_space) { exec_space.fence(); }

  static Event record_event(const ExecutionSpace& exec_space) {
    Event evt;
    evt.record(exec_space);
    return evt;
  }

  void add_dependency_to(
      const ExecutionSpace& /*exec_space*/) const { /* no-op */ }

  void wait() const { /* already fenced at record time */ }

  bool is_complete() const { return true; }
};

//============================================================================
// CUDA specialization — wraps CudaEvent
//============================================================================

#if defined(KOKKOS_ENABLE_CUDA)

template <>
class Event<Kokkos::Cuda> : public CudaEvent {
 public:
  using CudaEvent::CudaEvent;
};

#endif  // KOKKOS_ENABLE_CUDA

//============================================================================
// Free function: space_depends_on(event, exec_space)
//============================================================================

template <class ExecutionSpace>
void space_depends_on(const ExecutionSpace& exec_space,
                      const Event<ExecutionSpace>& event) {
  event.add_dependency_to(exec_space);
}

}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif
#endif  // KOKKOS_EXPERIMENTAL_EVENT_HPP
