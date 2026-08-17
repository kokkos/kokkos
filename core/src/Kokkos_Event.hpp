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
#include <Kokkos_Core.hpp>
#include <thread>

namespace Kokkos {
namespace Impl {
template <class ExecutionSpace>
struct EventResource {
  EventResource(const std::string& label_,
                const Kokkos::View<int, Kokkos::SharedHostPinnedSpace>& flag_,
                const ExecutionSpace& exec_)
      : label(label_), flag(flag_), exec(exec_) {}
  std::string label;
  Kokkos::View<int, Kokkos::SharedHostPinnedSpace> flag;
  ExecutionSpace exec;
};
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

// forward declare the class and the friend function space_depends_on
// so that we can make namespace qualified call work
template <Kokkos::ExecutionSpace Exec = DefaultExecutionSpace>
struct Event;

// Device-side dependency: the given execution space waits until the event
// has occured.
template <Kokkos::ExecutionSpace Exec>
void space_depends_on(const Exec& exec_space, const Event<Exec>& event);

template <Kokkos::ExecutionSpace Exec>
struct Event {
  using execution_space = Exec;

 private:
  using resource_t = Kokkos::Impl::EventResource<execution_space>;
  using handle_t   = std::shared_ptr<resource_t>;
  using flag_t     = Kokkos::View<int, Kokkos::SharedHostPinnedSpace>;

 public:
  Event(const std::string& label_)
      : m_handle(std::make_shared<resource_t>(
            label_, flag_t(std::string("Kokkos::Event::flag:" + label_)),
            execution_space())) {
    m_handle->flag() = 1;
  };

  Event(const std::string& label_, const execution_space& exec_space)
      : m_handle(std::make_shared<resource_t>(
            label_,
            Kokkos::View<int, Kokkos::SharedHostPinnedSpace>(
                std::string("Kokkos::Event::flag:") + label_),
            execution_space())) {
    record(exec_space);
  };

  // Create an event at the current spot in the execution space queue
  void record(const execution_space& exec_space) {
    m_handle->flag() = 0;
    m_handle->exec   = execution_space();
    auto flag        = m_handle->flag;
    Kokkos::parallel_for(
        std::string("Kokkos::Event::record:" + m_handle->label), 1,
        KOKKOS_LAMBDA(int) { flag() = 1; });
  }

  // Wait untile the even occurs
  void fence() const {
    while (m_handle->flag() != 1) std::this_thread::yield();
  }

  // Check whether the even has occured
  bool is_complete() const { return m_handle->flag() == 1; }

  const std::string& label() const { return m_handle->label; }

  // Enqueue a dependency on the event in an execution space instance
  friend void space_depends_on<execution_space>(
      const execution_space& exec_space, const Event<execution_space>& event);

 private:
  handle_t m_handle;
};

template <Kokkos::ExecutionSpace Exec>
void space_depends_on(const Exec& exec_space, const Event<Exec>& event) {
  // Only need to wait if its not the same execution space instance
  // Otherwise any work issues to
  if (exec_space != event.m_handle->exec) event.fence();
}
}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif
#endif  // KOKKOS_EVENT_HPP
