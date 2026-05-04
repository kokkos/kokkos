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
struct EventHandle<Kokkos::Cuda> {
  cudaEvent_t raw = nullptr;
  int device      = -1;

  EventHandle() : device(Kokkos::Cuda().cuda_device()) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(device));
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventCreateWithFlags(&raw, cudaEventDisableTiming));
  }

  explicit EventHandle(const Kokkos::Cuda& exec_space)
      : device(exec_space.cuda_device()) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(device));
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventCreateWithFlags(&raw, cudaEventDisableTiming));
  }

  ~EventHandle() {
    if (raw != nullptr) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(device));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(raw));
    }
  }

  EventHandle(const EventHandle&)            = delete;
  EventHandle& operator=(const EventHandle&) = delete;
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
      : m_handle(std::make_shared<Kokkos::Impl::EventHandle<Kokkos::Cuda>>()) {}

  Event(const Event&)            = default;
  Event& operator=(const Event&) = default;
  Event(Event&&)                 = default;
  Event& operator=(Event&&)      = default;
  ~Event()                       = default;

  Event(const Kokkos::Cuda& exec_space)
      : m_handle(std::make_shared<Kokkos::Impl::EventHandle<Kokkos::Cuda>>(
            exec_space)) {
    record(exec_space);
  }

  void record(const Kokkos::Cuda& exec_space) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(exec_space.cuda_device()));
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventRecord(m_handle->raw, exec_space.cuda_stream()));
  }

  void fence() const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_handle->device));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(m_handle->raw));
  }

  bool is_complete() const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_handle->device));
    cudaError_t err = cudaEventQuery(m_handle->raw);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    KOKKOS_IMPL_CUDA_SAFE_CALL(err);
    return false;
  }

  cudaEvent_t cuda_event() const noexcept { return m_handle->raw; }

 private:
  std::shared_ptr<Kokkos::Impl::EventHandle<Kokkos::Cuda>> m_handle;
};

/// CUDA: insert a stream wait for the recorded event (non-blocking on host).
inline void space_depends_on(const Kokkos::Cuda& exec_space,
                             const Event<Kokkos::Cuda>& event) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(exec_space.cuda_device()));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaStreamWaitEvent(exec_space.cuda_stream(), event.cuda_event(), 0));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA
#endif  // KOKKOS_CUDA_EVENT_HPP
