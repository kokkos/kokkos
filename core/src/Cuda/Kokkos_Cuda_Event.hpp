// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CUDA_EVENT_HPP
#define KOKKOS_CUDA_EVENT_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <Cuda/Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_Cuda_Error.hpp>

namespace Kokkos {
namespace Experimental {

/// A CudaEvent captures a point in a CUDA stream's execution timeline.
/// Other streams can then depend on that point without a full fence,
/// and the host can block on it selectively.

class CudaEvent {
 public:
  CudaEvent() { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreate(&m_event)); }

  ~CudaEvent() {
    if (m_event != nullptr) {
      (void)cudaEventDestroy(m_event);
    }
  }

  CudaEvent(const CudaEvent&)            = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept : m_event(other.m_event) {
    other.m_event = nullptr;
  }

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (m_event != nullptr) (void)cudaEventDestroy(m_event);
      m_event       = other.m_event;
      other.m_event = nullptr;
    }
    return *this;
  }

  //--------------------------------------------------------------------------
  // Recording
  //--------------------------------------------------------------------------

  /// All work submitted to exec_space before this call is captured
  /// by the event.
  void record(const Kokkos::Cuda& exec_space) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaEventRecord(m_event, exec_space.cuda_stream()));
  }

  /// \brief Create a new CudaEvent and immediately record it on
  ///        \p exec_space's stream.
  static CudaEvent record_event(const Kokkos::Cuda& exec_space) {
    CudaEvent evt;
    evt.record(exec_space);
    return evt;
  }

  //--------------------------------------------------------------------------
  // Dependency
  //--------------------------------------------------------------------------

  /// This is a GPU-side (non-blocking) dependency: the call returns
  /// immediately on the host, but any work subsequently submitted to
  /// exec_space will not start until the recorded event has finished.
  void add_dependency_to(const Kokkos::Cuda& exec_space) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaStreamWaitEvent(exec_space.cuda_stream(), m_event, 0));
  }

  //--------------------------------------------------------------------------
  // Host synchronisation
  //--------------------------------------------------------------------------

  ///  Block the calling host thread until this event completes.
  void wait() const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(m_event));
  }

  //--------------------------------------------------------------------------
  // Query
  //--------------------------------------------------------------------------

  /// Non-blocking query: has the recorded work completed?
  /// return true if all work captured by the event has finished.
  bool is_complete() const {
    cudaError_t err = cudaEventQuery(m_event);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    KOKKOS_IMPL_CUDA_SAFE_CALL(err);
    return false;  // unreachable, but silences compiler warnings
  }

  //--------------------------------------------------------------------------
  // Low-level access
  //--------------------------------------------------------------------------

  /// Return the underlying cudaEvent_t handle.
  cudaEvent_t cuda_event() const noexcept { return m_event; }

 private:
  cudaEvent_t m_event = nullptr;
};

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA
#endif  // KOKKOS_CUDA_EVENT_HPP
