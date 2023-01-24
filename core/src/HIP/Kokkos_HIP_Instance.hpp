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

#include <mutex>

namespace Kokkos {
namespace Impl {

struct HIPTraits {
#if defined(KOKKOS_ARCH_VEGA)
  static int constexpr WarpSize       = 64;
  static int constexpr WarpIndexMask  = 0x003f; /* hexadecimal for 63 */
  static int constexpr WarpIndexShift = 6;      /* WarpSize == 1 << WarpShift*/
#elif defined(KOKKOS_ARCH_NAVI)
  static int constexpr WarpSize       = 32;
  static int constexpr WarpIndexMask  = 0x001f; /* hexadecimal for 31 */
  static int constexpr WarpIndexShift = 5;      /* WarpSize == 1 << WarpShift*/
#endif
  static int constexpr ConservativeThreadsPerBlock =
      256;  // conservative fallback blocksize in case of spills
  static int constexpr MaxThreadsPerBlock =
      1024;  // the maximum we can fit in a block
  static int constexpr ConstantMemoryUsage        = 0x008000; /* 32k bytes */
  static int constexpr KernelArgumentLimit        = 0x001000; /*  4k bytes */
  static int constexpr ConstantMemoryUseThreshold = 0x000200; /* 512 bytes */
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
  inline static int m_hipArch                       = -1;
  inline static unsigned m_multiProcCount           = 0;
  inline static unsigned m_maxWarpCount             = 0;
  inline static std::array<size_type, 3> m_maxBlock = {0, 0, 0};
  inline static unsigned m_maxWavesPerCU            = 0;
  inline static unsigned m_maxSharedWords           = 0;
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
  std::int32_t *m_scratch_locks;

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

}  // namespace Impl

namespace Experimental {
// Partitioning an Execution Space: expects space and integer arguments for
// relative weight
//   Customization point for backends
//   Default behavior is to return the passed in instance

namespace Impl {
inline void create_HIP_instances(std::vector<HIP> &instances) {
  for (int s = 0; s < int(instances.size()); s++) {
    hipStream_t stream;
    KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamCreate(&stream));
    instances[s] = HIP(stream, true);
  }
}
}  // namespace Impl

template <class... Args>
std::vector<HIP> partition_space(const HIP &, Args...) {
  static_assert(
      (... && std::is_arithmetic_v<Args>),
      "Kokkos Error: partitioning arguments must be integers or floats");

  std::vector<HIP> instances(sizeof...(Args));
  Impl::create_HIP_instances(instances);
  return instances;
}

template <class T>
std::vector<HIP> partition_space(const HIP &, std::vector<T> &weights) {
  static_assert(
      std::is_arithmetic<T>::value,
      "Kokkos Error: partitioning arguments must be integers or floats");

  std::vector<HIP> instances(weights.size());
  Impl::create_HIP_instances(instances);
  return instances;
}
}  // namespace Experimental
}  // namespace Kokkos

#endif
