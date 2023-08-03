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

/*--------------------------------------------------------------------------*/

#ifndef KOKKOS_HIP_INSTANCE_HPP
#define KOKKOS_HIP_INSTANCE_HPP

#include <HIP/Kokkos_HIP_Space.hpp>
#include <HIP/Kokkos_HIP_Error.hpp>

#include <atomic>
#include <mutex>

namespace Kokkos {
namespace Impl {

struct HIPTraits {
#if defined(KOKKOS_ARCH_AMD_GFX906) || defined(KOKKOS_ARCH_AMD_GFX908) || \
    defined(KOKKOS_ARCH_AMD_GFX90A) || defined(KOKKOS_ARCH_AMD_GFX942)
  static constexpr int WarpSize       = 64;
  static constexpr int WarpIndexMask  = 0x003f; /* hexadecimal for 63 */
  static constexpr int WarpIndexShift = 6;      /* WarpSize == 1 << WarpShift*/
#elif defined(KOKKOS_ARCH_AMD_GFX1030) || defined(KOKKOS_ARCH_AMD_GFX1100)
  static constexpr int WarpSize       = 32;
  static constexpr int WarpIndexMask  = 0x001f; /* hexadecimal for 31 */
  static constexpr int WarpIndexShift = 5;      /* WarpSize == 1 << WarpShift*/
#endif
  static constexpr int ConservativeThreadsPerBlock =
      256;  // conservative fallback blocksize in case of spills
  static constexpr int MaxThreadsPerBlock =
      1024;  // the maximum we can fit in a block
  static constexpr int ConstantMemoryUsage        = 0x008000; /* 32k bytes */
  static constexpr int KernelArgumentLimit        = 0x001000; /*  4k bytes */
  static constexpr int ConstantMemoryUseThreshold = 0x000200; /* 512 bytes */
};

//----------------------------------------------------------------------------

HIP::size_type hip_internal_maximum_warp_count();
std::array<HIP::size_type, 3> hip_internal_maximum_grid_count();
HIP::size_type hip_internal_multiprocessor_count();

HIP::size_type *hip_internal_scratch_space(const HIP &instance,
                                           const std::size_t size);
HIP::size_type *hip_internal_scratch_flags(const HIP &instance,
                                           const std::size_t size);

//----------------------------------------------------------------------------

class HIPInternal {
 private:
  HIPInternal(const HIPInternal &);
  HIPInternal &operator=(const HIPInternal &);

 public:
  using size_type = ::Kokkos::HIP::size_type;

  inline static int m_hipDev                        = -1;
  inline static unsigned m_multiProcCount           = 0;
  inline static unsigned m_maxWarpCount             = 0;
  inline static std::array<size_type, 3> m_maxBlock = {0, 0, 0};
  inline static unsigned m_maxWavesPerCU            = 0;
  inline static int m_shmemPerSM                    = 0;
  inline static int m_maxShmemPerBlock              = 0;
  inline static int m_maxThreadsPerSM               = 0;

  inline static hipDeviceProp_t m_deviceProp;

  static int concurrency();

  // Scratch Spaces for Reductions
  std::size_t m_scratchSpaceCount          = 0;
  std::size_t m_scratchFlagsCount          = 0;
  mutable std::size_t m_scratchFunctorSize = 0;

  size_type *m_scratchSpace               = nullptr;
  size_type *m_scratchFlags               = nullptr;
  mutable size_type *m_scratchFunctor     = nullptr;
  mutable size_type *m_scratchFunctorHost = nullptr;
  inline static std::mutex scratchFunctorMutex;

  hipStream_t m_stream = nullptr;
  uint32_t m_instance_id =
      Kokkos::Tools::Experimental::Impl::idForInstance<HIP>(
          reinterpret_cast<uintptr_t>(this));
  bool m_manage_stream = false;

  // Team Scratch Level 1 Space
  int m_n_team_scratch                            = 10;
  mutable int64_t m_team_scratch_current_size[10] = {};
  mutable void *m_team_scratch_ptr[10]            = {};
  mutable std::atomic_int m_team_scratch_pool[10] = {};
  int32_t *m_scratch_locks                        = nullptr;
  size_t m_num_scratch_locks                      = 0;

  bool was_finalized = false;

  // FIXME_HIP: these want to be per-device, not per-stream...  use of 'static'
  // here will break once there are multiple devices though
  inline static unsigned long *constantMemHostStaging = nullptr;
  inline static hipEvent_t constantMemReusable        = nullptr;
  inline static std::mutex constantMemMutex;

  static HIPInternal &singleton();

  int verify_is_initialized(const char *const label) const;

  int is_initialized() const {
    return nullptr != m_scratchSpace && nullptr != m_scratchFlags;
  }

  void initialize(hipStream_t stream, bool manage_stream);
  void finalize();

  void print_configuration(std::ostream &) const;

  void fence() const;
  void fence(const std::string &) const;

  ~HIPInternal();

  HIPInternal() = default;

  // Using HIP API function/objects will be w.r.t. device 0 unless
  // hipSetDevice(device_id) is called with the correct device_id.
  // The correct device_id is stored in the variable
  // HIPInternal::m_hipDev set in HIP::impl_initialize(). It is not
  // sufficient to call hipSetDevice(m_hipDev) during HIP initialization
  // only, however, since if a user creates a new thread, that thread will be
  // given the default HIP env with device_id=0, causing errors when
  // device_id!=0 is requested by the user. To ensure against this, almost all
  // HIP API calls, as well as using hipStream_t variables, must be proceeded
  // by hipSetDevice(device_id).

  // This function sets device in HIP API to device requested at runtime (set in
  // m_hipDev).
  void set_hip_device() const {
    verify_is_initialized("set_hip_device");
    KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(m_hipDev));
  }

  // Return the class stream, optionally setting the device id.
  template <bool setHIPDevice = true>
  hipStream_t get_stream() const {
    if constexpr (setHIPDevice) set_hip_device();
    return m_stream;
  }

  // The following are wrappers for HIP API functions (C and C++ routines) which
  // set the correct device id directly before the HIP API call (unless
  // explicitly disabled by providing setHIPDevice=false template).
  // setHIPDevice=true should be used for all API calls which take a stream
  // unless it is guarenteed to be from a HIP instance with the correct device
  // set already (e.g., back-to-back HIP API calls in a single function). For
  // HIP API functions that take a stream, an optional input stream is
  // available. If no stream is given, the stream for the HIPInternal instance
  // is used. All HIP API calls should be wrapped in these interface functions
  // to ensure safety when using threads.

  // Helper function for selecting the correct input stream
  hipStream_t get_input_stream(hipStream_t s) const {
    return s == nullptr ? get_stream<false>() : s;
  }

  // HIP C API

  // Device Management
  template <bool setHIPDevice = true>
  hipError_t hip_device_get_attribute_wrapper(int *pi,
                                              hipDeviceAttribute_t attr,
                                              int deviceId) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipDeviceGetAttribute(pi, attr, deviceId);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_device_synchronize_wrapper() const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipDeviceSynchronize();
  }

  template <bool setHIPDevice = true>
  hipError_t hip_get_device_count_wrapper(int *count) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipGetDeviceCount(count);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_get_device_properties_wrapper(hipDeviceProp_t *prop,
                                               int deviceId) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipGetDeviceProperties(prop, deviceId);
  }

  // Error Handling
  template <bool setHIPDevice = true>
  const char *hip_get_error_name_wrapper(hipError_t hip_error) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipGetErrorName(hip_error);
  }

  template <bool setHIPDevice = true>
  const char *hip_get_error_string_wrapper(hipError_t hip_error) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipGetErrorString(hip_error);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_get_last_error_wrapper() const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipGetLastError();
  }

  // Event Management
  template <bool setHIPDevice = true>
  hipError_t hip_event_create_wrapper(hipEvent_t *event) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipEventCreate(event);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_event_destroy_wrapper(hipEvent_t event) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipEventDestroy(event);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_event_record_wrapper(hipEvent_t event,
                                      hipStream_t stream = nullptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipEventRecord(event, get_input_stream(stream));
  }

  template <bool setHIPDevice = true>
  hipError_t hip_event_synchronize_wrapper(hipEvent_t event) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipEventSynchronize(event);
  }

  // Memory Management
  template <bool setHIPDevice = true>
  hipError_t hip_free_wrapper(void *ptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipFree(ptr);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_host_free_wrapper(void *ptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipHostFree(ptr);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_host_malloc_wrapper(
      void **ptr, size_t size,
      unsigned int flags = hipHostMallocDefault) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipHostMalloc(ptr, size, flags);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_malloc_wrapper(void **ptr, size_t size) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipMalloc(ptr, size);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_malloc_managed_wrapper(
      void **dev_ptr, size_t size,
      unsigned int flags = hipMemAttachGlobal) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipMallocManaged(dev_ptr, size, flags);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_mem_advise_wrapper(const void *dev_ptr, size_t count,
                                    hipMemoryAdvise advice, int device) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipMemAdvise(dev_ptr, count, advice, device);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_memcpy_async_wrapper(void *dst, const void *src,
                                      size_t sizeBytes, hipMemcpyKind kind,
                                      hipStream_t stream = nullptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipMemcpyAsync(dst, src, sizeBytes, kind, get_input_stream(stream));
  }

  template <bool setHIPDevice = true>
  hipError_t hip_memcpy_to_symbol_async_wrapper(
      const void *symbol, const void *src, size_t sizeBytes, size_t offset,
      hipMemcpyKind kind, hipStream_t stream = nullptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind,
                                  get_input_stream(stream));
  }

  template <bool setHIPDevice = true>
  hipError_t hip_memset_wrapper(void *dst, int value, size_t sizeBytes) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipMemset(dst, value, sizeBytes);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_memset_async_wrapper(void *dst, int value, size_t sizeBytes,
                                      hipStream_t stream = nullptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipMemsetAsync(dst, value, sizeBytes, get_input_stream(stream));
  }

  // Module Management
  template <bool setHIPDevice = true>
  hipError_t hip_func_get_attributes_wrapper(struct hipFuncAttributes *attr,
                                             const void *func) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipFuncGetAttributes(attr, func);
  }

  // Stream Management
  template <bool setHIPDevice = true>
  hipError_t hip_stream_create_wrapper(hipStream_t *pStream) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipStreamCreate(pStream);
  }

  template <bool setHIPDevice = true>
  hipError_t hip_stream_destroy_wrapper(hipStream_t stream = nullptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipStreamDestroy(get_input_stream(stream));
  }

  template <bool setHIPDevice = true>
  hipError_t hip_stream_synchronize_wrapper(
      hipStream_t stream = nullptr) const {
    if constexpr (setHIPDevice) set_hip_device();
    return hipStreamSynchronize(get_input_stream(stream));
  }

  // Resizing of reduction related scratch spaces
  size_type *scratch_space(std::size_t const size);
  size_type *scratch_flags(std::size_t const size);
  size_type *stage_functor_for_execution(void const *driver,
                                         std::size_t const size) const;
  uint32_t impl_get_instance_id() const noexcept;
  int acquire_team_scratch_space();
  // Resizing of team level 1 scratch
  void *resize_team_scratch_space(int scratch_pool_id, std::int64_t bytes,
                                  bool force_shrink = false);
  void release_team_scratch_space(int scratch_pool_id);
};

void create_HIP_instances(std::vector<HIP> &instances);
}  // namespace Impl

namespace Experimental {
// Partitioning an Execution Space: expects space and integer arguments for
// relative weight
//   Customization point for backends
//   Default behavior is to return the passed in instance

template <class... Args>
std::vector<HIP> partition_space(const HIP &, Args...) {
  static_assert(
      (... && std::is_arithmetic_v<Args>),
      "Kokkos Error: partitioning arguments must be integers or floats");

  std::vector<HIP> instances(sizeof...(Args));
  Kokkos::Impl::create_HIP_instances(instances);
  return instances;
}

template <class T>
std::vector<HIP> partition_space(const HIP &, std::vector<T> const &weights) {
  static_assert(
      std::is_arithmetic<T>::value,
      "Kokkos Error: partitioning arguments must be integers or floats");

  // We only care about the number of instances to create and ignore weights
  // otherwise.
  std::vector<HIP> instances(weights.size());
  Kokkos::Impl::create_HIP_instances(instances);
  return instances;
}
}  // namespace Experimental
}  // namespace Kokkos

#endif
