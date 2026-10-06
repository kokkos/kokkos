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
#include <Kokkos_View.hpp>
#include <thread>

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

  EventResource(
      const std::string& label_,
      const Kokkos::View<uint64_t, Kokkos::SharedHostPinnedSpace>& flag_,
      const ExecutionSpace& exec_)
      : label(label_),
        counter(0),
        flag(flag_),
        exec(exec_),
        lock(std::mutex()) {}
  std::string label;

  uint64_t counter;
  Kokkos::View<uint64_t, Kokkos::SharedHostPinnedSpace> flag;

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

// FIXME: tried to use SeqCst for load and store but it didn't compile for CUDA
template <Kokkos::ExecutionSpace Exec>
struct Event {
  using execution_space = Exec;

 private:
  using resource_t = Kokkos::Impl::EventResource<execution_space>;
  using handle_t   = std::shared_ptr<resource_t>;
  using flag_t     = Kokkos::View<uint64_t, Kokkos::SharedHostPinnedSpace>;

 public:
  Event(const std::string& label_)
      : m_handle(std::make_shared<resource_t>(
            label_, flag_t(std::string("Kokkos::Event::flag:" + label_)),
            execution_space())) {
    desul::atomic_store(&(m_handle->flag()), uint64_t(0),
                        desul::MemoryOrderRelease(), desul::MemoryScopeNode());
    desul::atomic_store(&(m_handle->counter), uint64_t(0),
                        desul::MemoryOrderRelease(), desul::MemoryScopeNode());
  }

  Event(const std::string& label_, const execution_space& exec_space)
      : m_handle(std::make_shared<resource_t>(
            label_, flag_t(std::string("Kokkos::Event::flag:") + label_),
            execution_space())) {
    desul::atomic_store(&(m_handle->flag()), uint64_t(0),
                        desul::MemoryOrderRelease(), desul::MemoryScopeNode());
    desul::atomic_store(&(m_handle->counter), uint64_t(0),
                        desul::MemoryOrderRelease(), desul::MemoryScopeNode());
    record(exec_space);
  }

  // Create an event at the current spot in the execution space queue
  void record(const execution_space& exec_space) {
    m_handle->lock.lock();
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
    m_handle->lock.unlock();
  }

  // Wait until the event occurs
  void fence() const {
    m_handle->lock.lock();
    // Comparing for != is correct here, because of the lock mechanism
    // furthermore that actually means it will work with wrap around overflow
    // however unlikely that is considering the use of a 64bit integer
    while (desul::atomic_load(&m_handle->flag(), desul::MemoryOrderAcquire(),
                              desul::MemoryScopeNode()) !=
           desul::atomic_load(&m_handle->counter, desul::MemoryOrderAcquire(),
                              desul::MemoryScopeNode()))
      std::this_thread::yield();
    m_handle->lock.unlock();
  }

  // Check whether the event has occured
  bool is_complete() const {
    return desul::atomic_load(&m_handle->flag(), desul::MemoryOrderAcquire(),
                              desul::MemoryScopeNode()) ==
           desul::atomic_load(&m_handle->counter, desul::MemoryOrderAcquire(),
                              desul::MemoryScopeNode());
  }

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
  // Otherwise any work issues to the same instance will happen after the event
  if (exec_space != event.m_handle->exec) event.fence();
}
}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif
#endif  // KOKKOS_EVENT_HPP
