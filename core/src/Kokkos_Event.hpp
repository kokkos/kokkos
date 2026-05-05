// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Kokkos_Event.hpp
/// \brief Experimental event API for fine-grained stream dependencies.
///
/// Events capture a point in an execution space's asynchronous timeline.
/// They enable cross-stream dependencies without a full fence, and
/// selective host synchronisation.
///
/// API:
///   - space_depends_on(exec_space, event) — GPU-side dependency
///   (non-blocking on host)
///   - event.fence()                       — host-side blocking synchronisation
///   - event.is_complete()                 — non-blocking query
///
/// Currently only the CUDA backend provides a native implementation.
/// For other backends the fallback records a fence on record() and
/// space_depends_on / fence / is_complete are no-ops or trivially satisfied.

#ifndef KOKKOS_EVENT_HPP
#define KOKKOS_EVENT_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

namespace Kokkos {
namespace Impl {
template <class ExecutionSpace>
struct EventResource;
}  // namespace Impl

namespace Experimental {

//============================================================================
// Backend-agnostic Event — fallback for non-native-event backends
//============================================================================

/// Portable fallback event for backends without native event support.
///
/// On record(), a fence is issued so that subsequent space_depends_on()
/// and fence() are trivially satisfied.  This preserves correctness at
/// the cost of synchronisation -- the same trade-off existing Kokkos
/// code already makes.
///
/// Backends that provide a native implementation (e.g. CUDA) specialize
/// this template in backend-specific headers.
template <class ExecutionSpace = DefaultExecutionSpace>
class Event {
 public:
  Event() = default;

  void record(const ExecutionSpace& exec_space) { exec_space.fence(); }

  Event(const ExecutionSpace& exec_space) { record(exec_space); }

  void fence() const { /* already fenced at record time */
  }

  bool is_complete() const { return true; }
};

/// Device-side dependency: the given execution space waits until the event
/// is recorded. For the generic Event, this is a no-op; record()
/// already fenced the stream.
template <class ExecutionSpace>
void space_depends_on(const ExecutionSpace& /*exec_space*/,
                      const Event<ExecutionSpace>& /*event*/) {}

}  // namespace Experimental
}  // namespace Kokkos

#if defined(KOKKOS_ENABLE_CUDA)
#include <Cuda/Kokkos_Cuda_Event.hpp>
#endif

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif
#endif  // KOKKOS_EVENT_HPP
