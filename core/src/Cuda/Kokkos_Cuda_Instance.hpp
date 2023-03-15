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

#ifndef KOKKOS_CUDA_INSTANCE_HPP_
#define KOKKOS_CUDA_INSTANCE_HPP_

#include <vector>
#include <impl/Kokkos_Tools.hpp>
#include <atomic>
#include <Cuda/Kokkos_Cuda_Error.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// These functions fulfill the purpose of allowing to work around
// a suspected system software issue, or to check for race conditions.
// They are not currently a fully officially supported capability.
#ifdef KOKKOS_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
extern "C" void kokkos_impl_cuda_set_serial_execution(bool);
extern "C" bool kokkos_impl_cuda_use_serial_execution();
#endif

namespace Kokkos {
namespace Impl {

struct CudaTraits {
  static constexpr CudaSpace::size_type WarpSize = 32 /* 0x0020 */;
  static constexpr CudaSpace::size_type WarpIndexMask =
      0x001f; /* Mask for warpindex */
  static constexpr CudaSpace::size_type WarpIndexShift =
      5; /* WarpSize == 1 << WarpShift */

  static constexpr CudaSpace::size_type ConstantMemoryUsage =
      0x008000; /* 32k bytes */
  static constexpr CudaSpace::size_type ConstantMemoryCache =
      0x002000; /*  8k bytes */
  static constexpr CudaSpace::size_type KernelArgumentLimit =
      0x001000; /*  4k bytes */
  static constexpr CudaSpace::size_type MaxHierarchicalParallelism =
      1024; /* team_size * vector_length */
  using ConstantGlobalBufferType =
      unsigned long[ConstantMemoryUsage / sizeof(unsigned long)];

  static constexpr int ConstantMemoryUseThreshold = 0x000200 /* 512 bytes */;

  KOKKOS_INLINE_FUNCTION static CudaSpace::size_type warp_count(
      CudaSpace::size_type i) {
    return (i + WarpIndexMask) >> WarpIndexShift;
  }

  KOKKOS_INLINE_FUNCTION static CudaSpace::size_type warp_align(
      CudaSpace::size_type i) {
    constexpr CudaSpace::size_type Mask = ~WarpIndexMask;
    return (i + WarpIndexMask) & Mask;
  }
};

//----------------------------------------------------------------------------

CudaSpace::size_type cuda_internal_multiprocessor_count();
CudaSpace::size_type cuda_internal_maximum_warp_count();
std::array<CudaSpace::size_type, 3> cuda_internal_maximum_grid_count();
CudaSpace::size_type cuda_internal_maximum_shared_words();

CudaSpace::size_type cuda_internal_maximum_concurrent_block_count();

CudaSpace::size_type* cuda_internal_scratch_flags(const Cuda&,
                                                  const std::size_t size);
CudaSpace::size_type* cuda_internal_scratch_space(const Cuda&,
                                                  const std::size_t size);
CudaSpace::size_type* cuda_internal_scratch_unified(const Cuda&,
                                                    const std::size_t size);

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
namespace Kokkos {
namespace Impl {

class CudaInternal {
 private:
  CudaInternal(const CudaInternal&);
  CudaInternal& operator=(const CudaInternal&);
#ifdef KOKKOS_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
  static bool kokkos_impl_cuda_use_serial_execution_v;
#endif

 public:
  using size_type = Cuda::size_type;

  inline static int m_cudaDev = -1;

  // Device Properties
  inline static int m_cudaArch                      = -1;
  inline static unsigned m_multiProcCount           = 0;
  inline static unsigned m_maxWarpCount             = 0;
  inline static std::array<size_type, 3> m_maxBlock = {0, 0, 0};
  inline static unsigned m_maxSharedWords           = 0;
  inline static int m_shmemPerSM                    = 0;
  inline static int m_maxShmemPerBlock              = 0;
  inline static int m_maxBlocksPerSM                = 0;
  inline static int m_maxThreadsPerSM               = 0;
  inline static int m_maxThreadsPerBlock            = 0;
  static int concurrency();

  inline static cudaDeviceProp m_deviceProp;

  // Scratch Spaces for Reductions
  mutable std::size_t m_scratchSpaceCount;
  mutable std::size_t m_scratchFlagsCount;
  mutable std::size_t m_scratchUnifiedCount;
  mutable std::size_t m_scratchFunctorSize;

  inline static size_type m_scratchUnifiedSupported = 0;
  mutable size_type* m_scratchSpace;
  mutable size_type* m_scratchFlags;
  mutable size_type* m_scratchUnified;
  mutable size_type* m_scratchFunctor;
  cudaStream_t m_stream;
  uint32_t m_instance_id;
  bool m_manage_stream;

  // Team Scratch Level 1 Space
  int m_n_team_scratch = 10;
  mutable int64_t m_team_scratch_current_size[10];
  mutable void* m_team_scratch_ptr[10];
  mutable std::atomic_int m_team_scratch_pool[10];
  int32_t* m_scratch_locks;
  size_t m_num_scratch_locks;

  bool was_initialized = false;
  bool was_finalized   = false;

  // FIXME_CUDA: these want to be per-device, not per-stream...  use of 'static'
  //  here will break once there are multiple devices though
  inline static unsigned long* constantMemHostStaging = nullptr;
  inline static cudaEvent_t constantMemReusable       = nullptr;
  inline static std::mutex constantMemMutex;

  static CudaInternal& singleton();

  int verify_is_initialized(const char* const label) const;

  int is_initialized() const {
    return nullptr != m_scratchSpace && nullptr != m_scratchFlags;
  }

  void initialize(cudaStream_t stream, bool manage_stream);
  void finalize();

  void print_configuration(std::ostream&) const;

#ifdef KOKKOS_IMPL_DEBUG_CUDA_SERIAL_EXECUTION
  static bool cuda_use_serial_execution();
  static void cuda_set_serial_execution(bool);
#endif

  void fence(const std::string&) const;
  void fence() const;

  ~CudaInternal();

  CudaInternal()
      : m_scratchSpaceCount(0),
        m_scratchFlagsCount(0),
        m_scratchUnifiedCount(0),
        m_scratchFunctorSize(0),
        m_scratchSpace(nullptr),
        m_scratchFlags(nullptr),
        m_scratchUnified(nullptr),
        m_scratchFunctor(nullptr),
        m_stream(nullptr),
        m_instance_id(
            Kokkos::Tools::Experimental::Impl::idForInstance<Kokkos::Cuda>(
                reinterpret_cast<uintptr_t>(this))) {
    for (int i = 0; i < m_n_team_scratch; ++i) {
      m_team_scratch_current_size[i] = 0;
      m_team_scratch_ptr[i]          = nullptr;
      m_team_scratch_pool[i]         = 0;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////

  // Wrappers for Cuda API calls.
  void cuda_create_texture_object_api_wrapper(
      cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc,
      const cudaTextureDesc* pTexDesc,
      const cudaResourceViewDesc* pResViewDesc) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc));
  }

  void cuda_device_synchronize_api_wrapper() const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  void cuda_event_create_api_wrapper(cudaEvent_t* event) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreate(event));
  }

  void cuda_event_record_api_wrapper(cudaEvent_t event,
                                     cudaStream_t stream = nullptr) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(event, stream));
  }

  void cuda_event_synchronize_api_wrapper(cudaEvent_t event) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(event));
  }

  void cuda_free_api_wrapper(void* devPtr) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(devPtr));
  }

  void cuda_free_async_api_wrapper(
      [[maybe_unused]] void* devPtr,
      [[maybe_unused]] cudaStream_t hStream) const {
#if (CUDART_VERSION >= 11020)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeAsync(devPtr, hStream));
#else
    const std::string msg(
        "Kokkos::CudaInternal ERROR: "
        "cuda_free_async_api_wrapper() contains an internal "
        "cudaFreeAsync() API call, but "
        "Cuda version 11.2 or higher is required.");
    throw_runtime_exception(msg);
#endif
  }

  void cuda_free_host_api_wrapper(void* ptr) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeHost(ptr));
  }

  template <class FuncType>
  void cuda_func_get_attributes_api_wrapper(cudaFuncAttributes* attr,
                                            FuncType* entry) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFuncGetAttributes(attr, entry));
  }

  template <class FuncType>
  void cuda_func_set_attribute_api_wrapper(FuncType* entry,
                                           cudaFuncAttribute attr,
                                           int value) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFuncSetAttribute(entry, attr, value));
  }

  void cuda_get_last_error_api_wrapper(
      const bool remove_safe_call_macro = false) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    // There are cases where we don't want to error out here.
    if (remove_safe_call_macro)
      cudaGetLastError();
    else
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetLastError());
  }

  void cuda_get_pointer_attributes_api_wrapper(
      cudaPointerAttributes* attributes, const void* ptr) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaPointerGetAttributes(attributes, ptr));
  }

  void cuda_graph_add_empty_node_api_wrapper(
      cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
      const cudaGraphNode_t* pDependencies, size_t numDependencies) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphAddEmptyNode(
        pGraphNode, graph, pDependencies, numDependencies));
  }

  void cuda_graph_add_kernel_node_api_wrapper(
      cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
      const cudaGraphNode_t* pDependencies, size_t numDependencies,
      const cudaKernelNodeParams* pNodeParams) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGraphAddKernelNode(
        pGraphNode, graph, pDependencies, numDependencies, pNodeParams));
  }

  void cuda_malloc_api_wrapper(void** devPtr, size_t size) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(devPtr, size));
  }

  cudaError_t cuda_malloc_api_wrapper_return_error(void** devPtr,
                                                   size_t size) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    cudaError_t error_code = cudaMalloc(devPtr, size);
    return error_code;
  }

  cudaError_t cuda_malloc_async_api_wrapper_return_error(
      [[maybe_unused]] void** devPtr, [[maybe_unused]] size_t size,
      [[maybe_unused]] cudaStream_t hStream) const {
#if (defined(KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC) && CUDART_VERSION >= 11020)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    cudaError_t error_code = cudaMallocAsync(devPtr, size, hStream);
    return error_code;
#else
    const std::string msg(
        "Kokkos::CudaInternal ERROR: "
        "cuda_malloc_async_api_wrapper_return_error() contains an internal "
        "cudaMallocAsync() API call, but "
        "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC must be defined and Cuda version "
        "11.2 or higher is required.");
    throw_runtime_exception(msg);
#endif
  }

  void cuda_malloc_host_api_wrapper(void** ptr, size_t size) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMallocHost(ptr, size));
  }

  void cuda_mem_prefetch_async_api_wrapper(
      const void* devPtr, size_t count, int dstDevice,
      cudaStream_t stream = nullptr) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMemPrefetchAsync(devPtr, count, dstDevice, stream));
  }

  void cuda_memcpy_api_wrapper(void* dst, const void* src, size_t count,
                               cudaMemcpyKind kind) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpy(dst, src, count, kind));
  }

  void cuda_memcpy_async_api_wrapper(void* dst, const void* src, size_t count,
                                     cudaMemcpyKind kind,
                                     cudaStream_t stream) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemcpyAsync(dst, src, count, kind, stream));
  }

  void cuda_memcpy_to_symbol_async_api_wrapper(
      const void* symbol, const void* src, size_t count, size_t offset,
      cudaMemcpyKind kind, cudaStream_t stream = nullptr) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream));
  }

  void cuda_memset_api_wrapper(void* devPtr, int value, size_t count) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemset(devPtr, value, count));
  }

  void cuda_stream_create_api_wrapper(cudaStream_t* pStream) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(pStream));
  }

  void cuda_stream_synchronize_api_wrapper(cudaStream_t stream) const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  }

  ////////////////////////////////////////////////////////////////////////////////////////////

  // Resizing of reduction related scratch spaces
  size_type* scratch_space(const std::size_t size) const;
  size_type* scratch_flags(const std::size_t size) const;
  size_type* scratch_unified(const std::size_t size) const;
  size_type* scratch_functor(const std::size_t size) const;
  uint32_t impl_get_instance_id() const;
  int acquire_team_scratch_space();
  // Resizing of team level 1 scratch
  void* resize_team_scratch_space(int scratch_pool_id, std::int64_t bytes,
                                  bool force_shrink = false);
  void release_team_scratch_space(int scratch_pool_id);
};

}  // Namespace Impl

namespace Impl {

template <class DT, class... DP>
struct ZeroMemset<Kokkos::Cuda, DT, DP...> {
  ZeroMemset(const Kokkos::Cuda& exec_space_instance,
             const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemsetAsync(
        dst.data(), 0,
        dst.size() * sizeof(typename View<DT, DP...>::value_type),
        exec_space_instance.cuda_stream()));
  }

  ZeroMemset(const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    Kokkos::Impl::CudaInternal::singleton().cuda_memset_api_wrapper(
        reinterpret_cast<void*>(dst.data()), 0,
        dst.size() * sizeof(typename View<DT, DP...>::value_type));
  }
};
}  // namespace Impl

namespace Experimental {
// Partitioning an Execution Space: expects space and integer arguments for
// relative weight
//   Customization point for backends
//   Default behavior is to return the passed in instance

namespace Impl {
inline void create_Cuda_instances(std::vector<Cuda>& instances) {
  for (int s = 0; s < int(instances.size()); s++) {
    cudaStream_t stream;
    Kokkos::Impl::CudaInternal::singleton().cuda_stream_create_api_wrapper(
        &stream);
    instances[s] = Cuda(stream, true);
  }
}
}  // namespace Impl

template <class... Args>
std::vector<Cuda> partition_space(const Cuda&, Args...) {
  static_assert(
      (... && std::is_arithmetic_v<Args>),
      "Kokkos Error: partitioning arguments must be integers or floats");
  std::vector<Cuda> instances(sizeof...(Args));
  Impl::create_Cuda_instances(instances);
  return instances;
}

template <class T>
std::vector<Cuda> partition_space(const Cuda&, std::vector<T>& weights) {
  static_assert(
      std::is_arithmetic<T>::value,
      "Kokkos Error: partitioning arguments must be integers or floats");

  std::vector<Cuda> instances(weights.size());
  Impl::create_Cuda_instances(instances);
  return instances;
}
}  // namespace Experimental

}  // Namespace Kokkos
#endif
