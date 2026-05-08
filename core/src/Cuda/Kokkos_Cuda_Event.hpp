// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_CUDA_EVENT_HPP
#define KOKKOS_CUDA_EVENT_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#ifndef KOKKOS_EVENT_HPP
#include <Kokkos_Event.hpp>
#endif

#include <Cuda/Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_Cuda_Error.hpp>

#include <memory>

namespace Kokkos {
namespace Impl {

template <>
struct EventResource<Kokkos::Cuda> {
  cudaEvent_t m_event = nullptr;
  int m_cudaDev       = -1;

  EventResource() : EventResource(Kokkos::Cuda{}) {}

  explicit EventResource(const Kokkos::Cuda& exec_space)
      : m_cudaDev(exec_space.cuda_device()) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
  }

  ~EventResource() {
    if (m_event != nullptr) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(m_event));
    }
  }

  EventResource(const EventResource&)            = delete;
  EventResource& operator=(const EventResource&) = delete;
};

}  // namespace Impl

namespace Experimental {

//============================================================================
// CUDA specialization — native cudaEvent_t implementation
//============================================================================

/// CUDA specialization of Event.
///
/// Copyable: copies share the underlying cudaEvent_t via reference
/// counting.  The last copy standing destroys the event.  Re-recording
/// through any copy affects all copies (same semantics as sharing a
/// raw cudaEvent_t).
template <>
class Event<Kokkos::Cuda> {
 public:
  Event()
      : m_handle(
            std::make_shared<Kokkos::Impl::EventResource<Kokkos::Cuda>>()) {}

  Event(const Kokkos::Cuda& exec_space)
      : m_handle(std::make_shared<Kokkos::Impl::EventResource<Kokkos::Cuda>>(
            exec_space)) {
    record(exec_space);
  }

  void record(const Kokkos::Cuda& exec_space) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventRecord(m_handle->m_event, exec_space.cuda_stream()));
  }

  void fence() const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(m_handle->m_event));
  }

  bool is_complete() const {
    cudaError_t err = cudaEventQuery(m_handle->m_event);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    KOKKOS_IMPL_CUDA_SAFE_CALL(err);
    return false;
  }

  cudaEvent_t cuda_event() const noexcept { return m_handle->m_event; }

 private:
  std::shared_ptr<Kokkos::Impl::EventResource<Kokkos::Cuda>> m_handle;
};

/// CUDA: insert a stream wait for the recorded event (non-blocking on host).
inline void space_depends_on(const Kokkos::Cuda& exec_space,
                             const Event<Kokkos::Cuda>& event) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaStreamWaitEvent(exec_space.cuda_stream(), event.cuda_event(), 0));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA
#endif  // KOKKOS_CUDA_EVENT_HPP
