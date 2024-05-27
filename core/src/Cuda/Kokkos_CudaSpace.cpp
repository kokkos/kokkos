//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <Kokkos_Core.hpp>
#include <Cuda/Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_CudaSpace.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <unordered_set>

// #include <Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp>
#include <impl/Kokkos_Error.hpp>

#include <impl/Kokkos_Tools.hpp>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

cudaStream_t Kokkos::Impl::cuda_get_deep_copy_stream() {
  static cudaStream_t s = nullptr;
  if (s == nullptr) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        (CudaInternal::singleton().cuda_stream_create_wrapper(&s)));
  }
  return s;
}

const std::unique_ptr<Kokkos::Cuda> &Kokkos::Impl::cuda_get_deep_copy_space(
    bool initialize) {
  static std::unique_ptr<Cuda> space = nullptr;
  if (!space && initialize)
    space = std::make_unique<Cuda>(Kokkos::Impl::cuda_get_deep_copy_stream());
  return space;
}

namespace Kokkos {
namespace Impl {

namespace {

static std::atomic<int> num_uvm_allocations(0);

}  // namespace

void DeepCopyCuda(void *dst, const void *src, size_t n) {
  KOKKOS_IMPL_CUDA_SAFE_CALL((CudaInternal::singleton().cuda_memcpy_wrapper(
      dst, src, n, cudaMemcpyDefault)));
}

void DeepCopyAsyncCuda(const Cuda &instance, void *dst, const void *src,
                       size_t n) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      (instance.impl_internal_space_instance()->cuda_memcpy_async_wrapper(
          dst, src, n, cudaMemcpyDefault)));
}

void DeepCopyAsyncCuda(void *dst, const void *src, size_t n) {
  cudaStream_t s = cuda_get_deep_copy_stream();
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      (CudaInternal::singleton().cuda_memcpy_async_wrapper(
          dst, src, n, cudaMemcpyDefault, s)));
  Kokkos::Tools::Experimental::Impl::profile_fence_event<Kokkos::Cuda>(
      "Kokkos::Impl::DeepCopyAsyncCuda: Deep Copy Stream Sync",
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          DeepCopyResourceSynchronization,
      [&]() { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(s)); });
}

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
bool CudaUVMSpace::available() { return true; }
#endif

/*--------------------------------------------------------------------------*/

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
// The purpose of the following variable is to allow a state-based choice
// for pinning UVM allocations to the CPU. For now this is considered
// an experimental debugging capability - with the potential to work around
// some CUDA issues.
bool CudaUVMSpace::kokkos_impl_cuda_pin_uvm_to_host_v = false;

bool CudaUVMSpace::cuda_pin_uvm_to_host() {
  return CudaUVMSpace::kokkos_impl_cuda_pin_uvm_to_host_v;
}
void CudaUVMSpace::cuda_set_pin_uvm_to_host(bool val) {
  CudaUVMSpace::kokkos_impl_cuda_pin_uvm_to_host_v = val;
}
#endif
}  // namespace Kokkos

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
bool kokkos_impl_cuda_pin_uvm_to_host() {
  return Kokkos::CudaUVMSpace::cuda_pin_uvm_to_host();
}

void kokkos_impl_cuda_set_pin_uvm_to_host(bool val) {
  Kokkos::CudaUVMSpace::cuda_set_pin_uvm_to_host(val);
}
#endif

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

CudaSpace::CudaSpace()
    : m_device(Kokkos::Cuda().cuda_device()),
      m_stream(Kokkos::Cuda().cuda_stream()) {}
CudaSpace::CudaSpace(int device_id, cudaStream_t stream)
    : m_device(device_id), m_stream(stream) {}

CudaUVMSpace::CudaUVMSpace()
    : m_device(Kokkos::Cuda().cuda_device()),
      m_stream(Kokkos::Cuda().cuda_stream()) {}
CudaUVMSpace::CudaUVMSpace(int device_id, cudaStream_t stream)
    : m_device(device_id), m_stream(stream) {}

CudaHostPinnedSpace::CudaHostPinnedSpace()
    : m_device(Kokkos::Cuda().cuda_device()),
      m_stream(Kokkos::Cuda().cuda_stream()) {}
CudaHostPinnedSpace::CudaHostPinnedSpace(int device_id, cudaStream_t stream)
    : m_device(device_id), m_stream(stream) {}

size_t memory_threshold_g = 40000;  // 40 kB

//==============================================================================
// <editor-fold desc="allocate()"> {{{1

void *CudaSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *CudaSpace::allocate(const Cuda &exec_space, const char *arg_label,
                          const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  return impl_allocate(exec_space, arg_label, arg_alloc_size, arg_logical_size);
}
void *CudaSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

namespace {

#ifndef CUDART_VERSION
#error CUDART_VERSION undefined!
#elif (defined(KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC) && CUDART_VERSION >= 11020)

/**
 * If a device has been initialized we will drop it in the set
 * so we don't initialize it again
 */
std::unordered_set<int> _mem_pool_initialized;

/**
 * This is the initializer. It currently gets the device_id and stream
 * that impl_alloc gets
 *    @param device_id  The device id on which to alter the default mempool
 *    @param stream     A specific stream for cudaMallocAsync
 * 		@param error_code return any failure codes for cuda APIs
 *    @return  				  true for success, false for failure
 */
bool initializeMempool(const int device_id, const cudaStream_t stream,
                       cudaError_t &error_code) {
  error_code = cudaSuccess;

  std::cout << "Initializing Default Memory Pool for device " << device_id
            << "\n";

  // Interpret env var
  char *env_string = getenv("KOKKOS_CUDA_MEMPOOL_SIZE");
  if (!env_string) return true;  // Nothing to set, initialization is complete

  // Env var can be a decimal e.g. 2.4g
  double factor = 1;  // bytes

  // Interpret the string
  std::string mempool_size_string(env_string);
  char last_letter = mempool_size_string.back();

  // We can deal with the last letter being units.
  if (std::isalpha(last_letter)) {
    mempool_size_string.pop_back();
    char last_letter_cap = std::toupper(last_letter);
    switch (last_letter_cap) {
      case 'K': factor = 1024; break;
      case 'M': factor = 1024 * 1024; break;
      case 'G': factor = 1024 * 1024 * 1024; break;
      default:
        // Not permitted
        return false;
        break;
    }
  }

  // At this point we removed the last letter (or there was not a last letter)
  // convert what is left to a number.
  double requested_size = 0;

  // Handle exception in case the string is unconvertible
  try {
    requested_size = static_cast<size_t>(std::stod(mempool_size_string));
  } catch (...) {
    std::cerr << "Unable to convert " << mempool_size_string
              << " to a number\n";
    return false;
  }

  requested_size *= factor;
  size_t n_bytes = static_cast<size_t>(std::ceil(requested_size));
  if (!(n_bytes > 0)) return false;


  // We set up the default memory pool
  // This requires us to a) grab the default memory pool for the device
  // b) set a property to retain our desired amount of memory on a free
  // c) allocate our desired poolsize amount plus a little more
  // d) free the memory, ensuring the pool retains what we asked for
  // All the API calls need to return cudaSuccess or we fail

  // First get the default memory pool
  error_code = cudaSuccess;
  cudaMemPool_t default_mempool;  // The memory pool handle

  // Cuda API call to get the memory pool for device with device_id
  error_code = cudaDeviceGetDefaultMemPool(&default_mempool, device_id);
  if (error_code != cudaSuccess) return false;  // handle failure

  // Here we set the size of the retained memory in the pool after a
  // free by setting the cudaMemPoolAttrReleaseThresold property of the mempool
  error_code = cudaMemPoolSetAttribute(
      default_mempool, cudaMemPoolAttrReleaseThreshold, &n_bytes);
  if (error_code != cudaSuccess) return false;  // handle failure

  // Now we allocate the pool size amount + a little more (64 bytes is arbitrary
  // it just needs to be more than the retention size.
  void *dummy_ptr = nullptr;
  error_code      = cudaMallocAsync(&dummy_ptr, n_bytes + 64, stream);
  if (error_code != cudaSuccess) return false;  // handle failure

  // Now we free the memory, and our poolsize amount will be retained in the
  // pool
  error_code = cudaFreeAsync(dummy_ptr, stream);
  if (error_code != cudaSuccess) return false;  // handle failure

  // We are successful
  return true;
}

#endif

void *impl_allocate_common(const int device_id,
                           [[maybe_unused]] const cudaStream_t stream,
                           const char *arg_label, const size_t arg_alloc_size,
                           const size_t arg_logical_size,
                           const Kokkos::Tools::SpaceHandle arg_handle,
                           [[maybe_unused]] bool stream_sync_only) {
  void *ptr = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(device_id));

  cudaError_t error_code = cudaSuccess;
#ifndef CUDART_VERSION
#error CUDART_VERSION undefined!
#elif (defined(KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC) && CUDART_VERSION >= 11020)
  // Always use the Async. Don't use the 40K Lower limit - may well be arch
  // dependent.

  // Check the set of allocated devices, to check if our device is already
  // allocate it
  if (_mem_pool_initialized.find(device_id) == _mem_pool_initialized.end()) {
    // Not found: so initialize it
    //
    if (initializeMempool(device_id, stream, error_code)) {
      _mem_pool_initialized.insert(device_id);  // Success: add device to set
    } else {
      // Failure: report the error code with an Exception
      if (error_code != cudaSuccess) {
        cudaGetLastError();
        throw Experimental::CudaRawMemoryAllocationFailure(
            arg_alloc_size, error_code,
            Experimental::RawMemoryAllocationFailure::AllocationMechanism::
                CudaMallocAlignedPoolSetup);
      } else {
        // if the API calls all returned cudaSuccess, the strtod() could
        // have thrown an exception which we caught. In this case we
        // throw a dumb exception ehere
        throw "Memory Pool Size was incorrectly formatted or unsuitable size";
      }
    }
  }

  // Pool is definitely initialized (Potentially still zero bytes which is
  // default) do the originally intended allocation
  error_code = cudaMallocAsync(&ptr, arg_alloc_size, stream);
  if (error_code == cudaSuccess) {
    if (stream_sync_only) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    } else {
      Impl::cuda_device_synchronize(
          "Kokkos::Cuda: backend fence after async malloc");
    }
  } else {
    cudaGetLastError();
    throw Experimental::CudaRawMemoryAllocationFailure(
        arg_alloc_size, error_code,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::
            CudaMallocAligned);
  }
#else
  // If cudaMallocAsync is not available (either turned off, or too early CUDA
  // version) go with cudaMalloc()
  error_code = cudaMalloc(&ptr, arg_alloc_size);
  if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
    // This is the only way to clear the last error, which
    // we should do here since we're turning it into an
    // exception here
    cudaGetLastError();
    throw Experimental::CudaRawMemoryAllocationFailure(
        arg_alloc_size, error_code,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::
            CudaMalloc);
  }
#endif
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}
}  // namespace

void *CudaSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  return impl_allocate_common(m_device, m_stream, arg_label, arg_alloc_size,
                              arg_logical_size, arg_handle, false);
}

void *CudaSpace::impl_allocate(
    const Cuda &exec_space, const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  return impl_allocate_common(
      exec_space.cuda_device(), exec_space.cuda_stream(), arg_label,
      arg_alloc_size, arg_logical_size, arg_handle, true);
}

void *CudaUVMSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void *CudaUVMSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                             const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *CudaUVMSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void *ptr = nullptr;

  Cuda::impl_static_fence(
      "Kokkos::CudaUVMSpace::impl_allocate: Pre UVM Allocation");
  if (arg_alloc_size > 0) {
    Kokkos::Impl::num_uvm_allocations++;

    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_device));
    cudaError_t error_code =
        cudaMallocManaged(&ptr, arg_alloc_size, cudaMemAttachGlobal);

    if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
      // This is the only way to clear the last error, which
      // we should do here since we're turning it into an
      // exception here
      cudaGetLastError();
      throw Experimental::CudaRawMemoryAllocationFailure(
          arg_alloc_size, error_code,
          Experimental::RawMemoryAllocationFailure::AllocationMechanism::
              CudaMallocManaged);
    }

#ifdef KOKKOS_IMPL_DEBUG_CUDA_PIN_UVM_TO_HOST
    if (Kokkos::CudaUVMSpace::cuda_pin_uvm_to_host())
      KOKKOS_IMPL_CUDA_SAFE_CALL(
          cudaMemAdvise(ptr, arg_alloc_size, cudaMemAdviseSetPreferredLocation,
                        cudaCpuDeviceId));
#endif
  }
  Cuda::impl_static_fence(
      "Kokkos::CudaUVMSpace::impl_allocate: Post UVM Allocation");
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}
void *CudaHostPinnedSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void *CudaHostPinnedSpace::allocate(const char *arg_label,
                                    const size_t arg_alloc_size,
                                    const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}
void *CudaHostPinnedSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  void *ptr = nullptr;

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_device));
  cudaError_t error_code =
      cudaHostAlloc(&ptr, arg_alloc_size, cudaHostAllocDefault);
  if (error_code != cudaSuccess) {  // TODO tag as unlikely branch
    // This is the only way to clear the last error, which
    // we should do here since we're turning it into an
    // exception here
    cudaGetLastError();
    throw Experimental::CudaRawMemoryAllocationFailure(
        arg_alloc_size, error_code,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::
            CudaHostAlloc);
  }
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}

// </editor-fold> end allocate() }}}1
//==============================================================================
void CudaSpace::deallocate(void *const arg_alloc_ptr,
                           const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void CudaSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                           const size_t arg_alloc_size,
                           const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void CudaSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  try {
#ifndef CUDART_VERSION
#error CUDART_VERSION undefined!
#elif (defined(KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC) && CUDART_VERSION >= 11020)
    if (arg_alloc_size >= memory_threshold_g) {
      Impl::cuda_device_synchronize(
          "Kokkos::Cuda: backend fence before async free");
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_device));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeAsync(arg_alloc_ptr, m_stream));
      Impl::cuda_device_synchronize(
          "Kokkos::Cuda: backend fence after async free");
    } else {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_device));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(arg_alloc_ptr));
    }
#else
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_device));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(arg_alloc_ptr));
#endif
  } catch (...) {
  }
}
void CudaUVMSpace::deallocate(void *const arg_alloc_ptr,
                              const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void CudaUVMSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                              const size_t arg_alloc_size

                              ,
                              const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void CudaUVMSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  Cuda::impl_static_fence(
      "Kokkos::CudaUVMSpace::impl_deallocate: Pre UVM Deallocation");
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  try {
    if (arg_alloc_ptr != nullptr) {
      Kokkos::Impl::num_uvm_allocations--;
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_device));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(arg_alloc_ptr));
    }
  } catch (...) {
  }
  Cuda::impl_static_fence(
      "Kokkos::CudaUVMSpace::impl_deallocate: Post UVM Deallocation");
}

void CudaHostPinnedSpace::deallocate(void *const arg_alloc_ptr,
                                     const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void CudaHostPinnedSpace::deallocate(const char *arg_label,
                                     void *const arg_alloc_ptr,
                                     const size_t arg_alloc_size,
                                     const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void CudaHostPinnedSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  try {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_device));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeHost(arg_alloc_ptr));
  } catch (...) {
  }
}

}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

void cuda_prefetch_pointer(const Cuda &space, const void *ptr, size_t bytes,
                           bool to_device) {
  if ((ptr == nullptr) || (bytes == 0)) return;
  cudaPointerAttributes attr;
  KOKKOS_IMPL_CUDA_SAFE_CALL((
      space.impl_internal_space_instance()->cuda_pointer_get_attributes_wrapper(
          &attr, ptr)));
  // I measured this and it turns out prefetching towards the host slows
  // DualView syncs down. Probably because the latency is not too bad in the
  // first place for the pull down. If we want to change that provde
  // cudaCpuDeviceId as the device if to_device is false
  bool is_managed = attr.type == cudaMemoryTypeManaged;
  if (to_device && is_managed &&
      space.cuda_device_prop().concurrentManagedAccess) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        (space.impl_internal_space_instance()->cuda_mem_prefetch_async_wrapper(
            ptr, bytes, space.cuda_device())));
  }
}

}  // namespace Impl
}  // namespace Kokkos

//==============================================================================
// <editor-fold desc="Explicit instantiations of CRTP Base classes"> {{{1

#include <impl/Kokkos_SharedAlloc_timpl.hpp>

KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::CudaSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::CudaUVMSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    Kokkos::CudaHostPinnedSpace);

// </editor-fold> end Explicit instantiations of CRTP Base classes }}}1
//==============================================================================

#else
void KOKKOS_CORE_SRC_CUDA_CUDASPACE_PREVENT_LINK_ERROR() {}
#endif  // KOKKOS_ENABLE_CUDA
