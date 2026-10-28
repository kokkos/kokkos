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

#include <Kokkos_Event.hpp>
#include <Kokkos_Assert.hpp>

#include <Cuda/Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_Cuda_Error.hpp>
#include <Cuda/Kokkos_Cuda_Instance.hpp>

#include <memory>
#include <string>

namespace Kokkos {
namespace Impl {

struct CudaEventResource {
  std::string m_label = "unknown";
  cudaEvent_t m_event = nullptr;
  int m_cudaDev       = -1;

  explicit CudaEventResource(const std::string& label,
                             const Kokkos::Cuda& exec_space)
      : m_label(label), m_cudaDev(exec_space.cuda_device()) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(exec_space.impl_internal_space_instance()
                                   ->cuda_event_create_with_flags_wrapper(
                                       &m_event, cudaEventDisableTiming));
  }

  ~CudaEventResource() {
    if (m_event != nullptr) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(m_event));
    }
  }

  CudaEventResource(const CudaEventResource&)            = delete;
  CudaEventResource& operator=(const CudaEventResource&) = delete;
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
/// After a move, an Event may be destroyed or assigned a valid Event.
/// Other operations require a valid handle.
template <>
class Event<Kokkos::Cuda> {
 public:
  explicit Event(const std::string& label)
      : m_handle(std::make_shared<Kokkos::Impl::CudaEventResource>(
            label, Kokkos::Cuda())) {}

  Event(const std::string& label, const Kokkos::Cuda& exec_space)
      : m_handle(std::make_shared<Kokkos::Impl::CudaEventResource>(
            label, exec_space)) {
    record(exec_space);
  }

  void record(const Kokkos::Cuda& exec_space) {
    KOKKOS_EXPECTS(bool(m_handle));
    KOKKOS_EXPECTS(exec_space.cuda_device() == m_handle->m_cudaDev);
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        exec_space.impl_internal_space_instance()->cuda_event_record_wrapper(
            m_handle->m_event));
  }

  void fence() const {
    KOKKOS_EXPECTS(bool(m_handle));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(m_handle->m_event));
  }

  bool is_complete() const {
    KOKKOS_EXPECTS(bool(m_handle));
    cudaError_t err = cudaEventQuery(m_handle->m_event);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    KOKKOS_IMPL_CUDA_SAFE_CALL(err);
    return false;
  }

  const std::string& label() const {
    KOKKOS_EXPECTS(bool(m_handle));
    return m_handle->m_label;
  }
  cudaEvent_t cuda_event() const noexcept {
    KOKKOS_EXPECTS(bool(m_handle));
    return m_handle->m_event;
  }

 private:
  std::shared_ptr<Kokkos::Impl::CudaEventResource> m_handle;
};

/// CUDA: insert a stream wait for the recorded event (non-blocking on host).
template <>
inline void space_depends_on<Kokkos::Cuda, Kokkos::Cuda>(
    const Kokkos::Cuda& exec_space, const Event<Kokkos::Cuda>& event) {
  const auto cuda_event = event.cuda_event();
  exec_space.impl_internal_space_instance()->set_cuda_device();
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaStreamWaitEvent(exec_space.cuda_stream(), cuda_event, 0));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA
#endif  // KOKKOS_CUDA_EVENT_HPP
