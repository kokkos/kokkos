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
///   - space_depends_on(exec_space, event) — GPU-side dependency
///   (non-blocking on host)
///   - event.fence()                       — host-side blocking synchronisation
///   - event.is_complete()                 — non-blocking query
///
/// Currently only the CUDA backend provides a native implementation.
/// For other backends the fallback records a fence on record() and
/// space_depends_on / fence / is_complete are no-ops or trivially satisfied.

#ifndef KOKKOS_EXPERIMENTAL_EVENT_HPP
#define KOKKOS_EXPERIMENTAL_EVENT_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <memory>

#if defined(KOKKOS_ENABLE_CUDA)
#include <Cuda/Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_Cuda_Error.hpp>
#endif

namespace Kokkos {
namespace Experimental {

//============================================================================
// Backend-agnostic Event — fallback for non-native-event backends
//============================================================================

///  Portable fallback event for backends without native event support.
///
/// On record(), a fence is issued so that subsequent space_depends_on()
/// and fence() are trivially satisfied.  This preserves correctness at
/// the cost of synchronisation -- the same trade-off existing Kokkos
/// code already makes.
///
/// Backends that provide a native implementation (e.g. CUDA) specialize
/// this template below.
template <class ExecutionSpace = DefaultExecutionSpace>
class Event {
 public:
  Event() = default;

  void record(const ExecutionSpace& exec_space) { exec_space.fence(); }

  Event(const ExecutionSpace& exec_space) { record(exec_space); }

  void fence() const { /* already fenced at record time */ }

  bool is_complete() const { return true; }
};

/// Device-side dependency: the given execution space waits until the event
/// is recorded. For the generic Event, this is a no-op; record()
/// already fenced the stream.
template <class ExecutionSpace>
void space_depends_on(const ExecutionSpace& /*exec_space*/,
                      const Event<ExecutionSpace>& /*event*/) {}

//============================================================================
// CUDA specialization — native cudaEvent_t implementation
//============================================================================

#if defined(KOKKOS_ENABLE_CUDA)

/// CUDA specialization of Event.
///
/// Copyable: copies share the underlying cudaEvent_t via reference
/// counting.  The last copy standing destroys the event.  Re-recording
/// through any copy affects all copies (same semantics as sharing a
/// raw cudaEvent_t).
template <>
class Event<Kokkos::Cuda> {
  struct CudaEventHandle {
    cudaEvent_t raw;
    CudaEventHandle() {
      KOKKOS_IMPL_CUDA_SAFE_CALL(
          cudaEventCreateWithFlags(&raw, cudaEventDisableTiming));
    }
    ~CudaEventHandle() { (void)cudaEventDestroy(raw); }
    CudaEventHandle(const CudaEventHandle&)            = delete;
    CudaEventHandle& operator=(const CudaEventHandle&) = delete;
  };

 public:
  Event() : m_handle(std::make_shared<CudaEventHandle>()) {}

  Event(const Event&)            = default;
  Event& operator=(const Event&) = default;
  Event(Event&&)                 = default;
  Event& operator=(Event&&)      = default;
  ~Event()                       = default;

  Event(const Kokkos::Cuda& exec_space)
      : m_handle(std::make_shared<CudaEventHandle>()) {
    record(exec_space);
  }

  void record(const Kokkos::Cuda& exec_space) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventRecord(m_handle->raw, exec_space.cuda_stream()));
  }

  void fence() const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(m_handle->raw));
  }

  bool is_complete() const {
    cudaError_t err = cudaEventQuery(m_handle->raw);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    KOKKOS_IMPL_CUDA_SAFE_CALL(err);
    return false;
  }

  cudaEvent_t cuda_event() const noexcept { return m_handle->raw; }

 private:
  std::shared_ptr<CudaEventHandle> m_handle;
};

/// CUDA: insert a stream wait for the recorded event (non-blocking on host).
inline void space_depends_on(const Kokkos::Cuda& exec_space,
                             const Event<Kokkos::Cuda>& event) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaStreamWaitEvent(exec_space.cuda_stream(), event.cuda_event(), 0));
}

#endif  // KOKKOS_ENABLE_CUDA

}  // namespace Experimental
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_EVENT
#endif
#endif  // KOKKOS_EXPERIMENTAL_EVENT_HPP
