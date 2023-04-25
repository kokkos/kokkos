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

  // Using cudaAPI function/objects will be w.r.t. device 0 unless
  // cudaSetDevice(device_id) is called with the correct device_id.
  // The correct device_id is stored in the static variable
  // CudaInternal::m_cudaDev set in Cuda::impl_initialize(). It is not
  // sufficient to call cudaSetDevice(m_cudaDev) during cuda initialization
  // only, however, since if a user creates a new thread, that thread will be
  // given the default cuda env with device_id=0, causing errors when
  // device_id!=0 is requested by the user. To ensure against this, almost all
  // cudaAPI calls, as well as using cudaStream_t variables, must be proceeded
  // by cudaSetDevice(device_id).

  // This function sets device in cudaAPI to device requested at runtime (set in
  // m_cudaDev).
  void set_cuda_device() const {
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(m_cudaDev));
  }

  // The following takes a cudaAPI function reference and its arguments as
  // input, and either wraps the call in KOKKOS_IMPL_CUDA_SAFE_CALL() (if no
  // return value needed, throws if error code is given) or returns the result
  // of the function. If template setCudaDevice is not given, defaults to true.
  // Only if the call is guarenteed to be in a cuda instance with the correct
  // device should setCudaDevice=false. All cudaAPI calls should be wrapped in
  // these interface functions to ensure saftey when using threads.

  // Some cudaAPI functions take template args which are in addition to its
  // InputArgs. In this case the functions referenced must include the instance
  // requested. For an example, see calls to cudaCreateChannelDesc<T>() in
  // SharedAllocationRecord::attach_texture_object(). At times the InputArgs may
  // not be deduced correctly by certain compilers. In most cases, explicitly
  // providing them solves this issue. For an example, see call to
  // cudaMemset(void*, int, size_t) in CudaInternal::scratch_flags(). If the
  // compiler still does not recognize InputArgs, some input vars may need to be
  // cast to their correct type. For an example, see call to cudaMalloc() in
  // CudaInternal::initialize().
  template <bool setCudaDevice, typename... InputArgs>
  void cuda_api_interface_safe_call(
      cudaError_t (*cuda_api_function)(InputArgs...),
      InputArgs... cuda_api_input) const {
    if (setCudaDevice) set_cuda_device();
    KOKKOS_IMPL_CUDA_SAFE_CALL(cuda_api_function(cuda_api_input...));
  }

  template <typename... InputArgs>
  void cuda_api_interface_safe_call(
      cudaError_t (*cuda_api_function)(InputArgs...),
      InputArgs... cuda_api_input) const {
    cuda_api_interface_safe_call<true, InputArgs...>(cuda_api_function,
                                                     cuda_api_input...);
  }

  template <bool setCudaDevice, typename ReturnType, typename... InputArgs>
  ReturnType cuda_api_interface_return(
      ReturnType (*cuda_api_function)(InputArgs...),
      InputArgs... cuda_api_input) const {
    if (setCudaDevice) set_cuda_device();
    return cuda_api_function(cuda_api_input...);
  }

  template <typename ReturnType, typename... InputArgs>
  ReturnType cuda_api_interface_return(
      ReturnType (*cuda_api_function)(InputArgs...),
      InputArgs... cuda_api_input) const {
    return cuda_api_interface_return<true, ReturnType, InputArgs...>(
        cuda_api_function, cuda_api_input...);
  }

  // Using the m_stream variable can also cause issues when device_id!=0.
  template <bool setCudaDevice = true>
  cudaStream_t get_stream() const {
    if (setCudaDevice) set_cuda_device();
    return m_stream;
  }

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
    Kokkos::Impl::CudaInternal::singleton()
        .cuda_api_interface_safe_call<void*, int, size_t, cudaStream_t>(
            &cudaMemsetAsync, dst.data(), 0,
            dst.size() * sizeof(typename View<DT, DP...>::value_type),
            exec_space_instance.cuda_stream());
  }

  ZeroMemset(const View<DT, DP...>& dst,
             typename View<DT, DP...>::const_value_type&) {
    Kokkos::Impl::CudaInternal::singleton()
        .cuda_api_interface_safe_call<void*, int, size_t>(
            &cudaMemset, dst.data(), 0,
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
    Kokkos::Impl::CudaInternal::singleton()
        .cuda_api_interface_safe_call<cudaStream_t*>(&cudaStreamCreate,
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
