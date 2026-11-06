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
///   - space_depends_on(exec_space, event) — order work after the event;
///     some specializations (e.g. CUDA) are non-blocking on the host, while
///     fallback and cross-backend dependencies may block the host
///   - event.fence()                       — host-side blocking synchronisation
///   - event.is_complete()                 — non-blocking query
///
/// Currently only the CUDA backend provides a native implementation.
/// Other backends enqueue a completion marker on record(). Cross-backend
/// dependencies synchronize the event on the host before returning.

#ifndef KOKKOS_EVENT_HPP
#define KOKKOS_EVENT_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_View.hpp>
#include <mutex>
#include <thread>
#include <type_traits>

namespace Kokkos {

namespace Impl {

template <class ExecutionSpace>
struct EventResource {
#ifdef KOKKOS_ENABLE_NEXTSILICON
  using flag_memory_space_t =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::NextSilicon>,
                         Kokkos::SharedSpace, Kokkos::SharedHostPinnedSpace>;
#else
  using flag_memory_space_t = Kokkos::SharedHostPinnedSpace;
#endif

  using flag_t =
      Kokkos::View<uint64_t,
                   Kokkos::Device<ExecutionSpace, flag_memory_space_t>>;

  EventResource(const std::string& label_, const flag_t& flag_,
                const ExecutionSpace& exec_)
      : label(label_), counter(0), flag(flag_), exec(exec_), lock() {}
  std::string label;

  uint64_t counter;
  flag_t flag;

  ExecutionSpace exec;
  std::mutex lock;
};
}  // namespace Impl

namespace Experimental {

//============================================================================
// Backend-agnostic Event — fallback for non-native-event backends
//============================================================================

/// Portable fallback event for backends without native event support.
///
/// record() enqueues a completion marker and fence() waits on the host.
/// A dependency on the same instance relies on that instance's ordering;
/// dependencies on other instances or backends wait for event completion.
///
/// Backends that provide a native implementation (e.g. CUDA) specialize
/// this template in backend-specific headers.
///
/// After a move, an Event may be destroyed or assigned a valid Event.
/// Other operations require a valid handle.

// forward declare the class and the friend function space_depends_on
// so that we can make namespace qualified call work
template <Kokkos::ExecutionSpace Exec = DefaultExecutionSpace>
class Event;

// Order subsequent work in the given execution space after the event.
// Use return-type SFINAE to avoid constrained-friend crashes in older
// Intel/LLVM compilers.
template <class WaiterExec, class EventExec>
std::enable_if_t<Kokkos::ExecutionSpace<WaiterExec> &&
                 Kokkos::ExecutionSpace<EventExec>>
space_depends_on(const WaiterExec& exec_space, const Event<EventExec>& event);

// FIXME: tried to use SeqCst for load and store but it didn't compile for CUDA
template <Kokkos::ExecutionSpace Exec>
class Event {
 public:
  using execution_space = Exec;

 private:
  using resource_t = Kokkos::Impl::EventResource<execution_space>;
  using handle_t   = std::shared_ptr<resource_t>;
  using flag_t     = typename resource_t::flag_t;

 public:
  explicit Event(const std::string& label_)
      : m_handle(std::make_shared<resource_t>(
            label_, flag_t(std::string("Kokkos::Event::flag:" + label_)),
            execution_space())) {}

  Event(const std::string& label_, const execution_space& exec_space)
      : m_handle(std::make_shared<resource_t>(
            label_, flag_t(std::string("Kokkos::Event::flag:") + label_),
            execution_space())) {
    record(exec_space);
  }

  // Create an event at the current spot in the execution space queue
  void record(const execution_space& exec_space) {
    KOKKOS_EXPECTS(bool(m_handle));
    std::lock_guard<std::mutex> guard(m_handle->lock);
    m_handle->exec = exec_space;
    desul::atomic_inc(&m_handle->counter, desul::MemoryOrderSeqCst(),
                      desul::MemoryScopeNode());
    auto flag = m_handle->flag;
    Kokkos::parallel_for(
        std::string("Kokkos::Event::record:" + m_handle->label),
        Kokkos::RangePolicy(exec_space, 0, 1), KOKKOS_LAMBDA(int) {
          desul::atomic_inc(&(flag()), desul::MemoryOrderSeqCst(),
                            desul::MemoryScopeNode());
        });
  }

  // Wait until the event occurs
  void fence() const {
    KOKKOS_EXPECTS(bool(m_handle));
    std::lock_guard<std::mutex> guard(m_handle->lock);
    while (!is_complete()) std::this_thread::yield();
  }

  // Check whether the event has occured.
  bool is_complete() const {
    KOKKOS_EXPECTS(bool(m_handle));
    return desul::atomic_load(&m_handle->flag(), desul::MemoryOrderAcquire(),
                              desul::MemoryScopeNode()) ==
           desul::atomic_load(&m_handle->counter, desul::MemoryOrderAcquire(),
                              desul::MemoryScopeNode());
  }

  const std::string& label() const {
    KOKKOS_EXPECTS(bool(m_handle));
    return m_handle->label;
  }

  // Enqueue a dependency on the event in an execution space instance
  friend void space_depends_on<execution_space, execution_space>(
      const execution_space& exec_space, const Event<execution_space>& event);

 private:
  handle_t m_handle;
};

template <class WaiterExec, class EventExec>
std::enable_if_t<Kokkos::ExecutionSpace<WaiterExec> &&
                 Kokkos::ExecutionSpace<EventExec>>
space_depends_on(const WaiterExec& exec_space, const Event<EventExec>& event) {
  if constexpr (std::is_same_v<WaiterExec, EventExec>) {
    KOKKOS_EXPECTS(bool(event.m_handle));
    // Only compare instances of the same backend. Matching instances already
    // order subsequent work after the event.
    std::lock_guard<std::mutex> guard(event.m_handle->lock);
    if (exec_space != event.m_handle->exec) {
      while (!event.is_complete()) std::this_thread::yield();
    }
  } else {
    event.fence();
  }
}
}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif
#endif  // KOKKOS_EVENT_HPP
