/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_HIP_KERNEL_LAUNCH_HPP
#define KOKKOS_HIP_KERNEL_LAUNCH_HPP

#include <Kokkos_Macros.hpp>

#if defined(__HIPCC__)

#include <HIP/Kokkos_HIP_Error.hpp>
#include <HIP/Kokkos_HIP_Instance.hpp>
#include <Kokkos_HIP_Space.hpp>
#include <HIP/Kokkos_HIP_Locks.hpp>

// Must use global variable on the device with HIP-Clang
#ifdef __HIP__
__device__ __constant__ unsigned long kokkos_impl_hip_constant_memory_buffer
    [Kokkos::Experimental::Impl::HIPTraits::ConstantMemoryUsage /
     sizeof(unsigned long)];
#endif

namespace Kokkos {
namespace Experimental {
template <typename T>
inline __device__ T *kokkos_impl_hip_shared_memory() {
  extern __shared__ Kokkos::Experimental::HIPSpace::size_type sh[];
  return (T *)sh;
}
}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename DriverType>
__global__ static void hip_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_hip_constant_memory_buffer));
  driver();
}

template <typename DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ __launch_bounds__(
    maxTperB, minBperSM) static void hip_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_hip_constant_memory_buffer));

  driver->operator()();
}

template <class DriverType>
__global__ static void hip_parallel_launch_local_memory(
    const DriverType *driver) {
  driver->operator()();
}

template <class DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ __launch_bounds__(
    maxTperB,
    minBperSM) static void hip_parallel_launch_local_memory(const DriverType
                                                                *driver) {
  driver->operator()();
}

enum class HIPLaunchMechanism : unsigned {
  Default        = 0,
  ConstantMemory = 1,
  GlobalMemory   = 2,
  LocalMemory    = 4
};

enum BlockType : unsigned { Max = 0, Preferred = 1 };

constexpr inline HIPLaunchMechanism operator|(HIPLaunchMechanism p1,
                                              HIPLaunchMechanism p2) {
  return static_cast<HIPLaunchMechanism>(static_cast<unsigned>(p1) |
                                         static_cast<unsigned>(p2));
}
constexpr inline HIPLaunchMechanism operator&(HIPLaunchMechanism p1,
                                              HIPLaunchMechanism p2) {
  return static_cast<HIPLaunchMechanism>(static_cast<unsigned>(p1) &
                                         static_cast<unsigned>(p2));
}

template <HIPLaunchMechanism l>
struct HIPDispatchProperties {
  HIPLaunchMechanism launch_mechanism = l;
};

template <typename DriverType, typename LaunchBounds,
          HIPLaunchMechanism LaunchMechanism>
struct HIPParallelLaunchKernelFunc;

template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    HIPLaunchMechanism::LocalMemory> {
  static constexpr auto default_launchbounds() { return false; }

  static auto get_kernel_func() {
    return hip_parallel_launch_local_memory<DriverType, MaxThreadsPerBlock,
                                            MinBlocksPerSM>;
  }

  static unsigned int get_scratch_size() {
    return get_hip_func_attributes().localSizeBytes;
  }

  static hipFuncAttributes get_hip_func_attributes() {
    static hipFuncAttributes attr = []() {
      hipFuncAttributes attr;
      HIP_SAFE_CALL(hipFuncGetAttributes(
          &attr, reinterpret_cast<void const *>(get_kernel_func())));
      return attr;
    }();
    return attr;
  }
};

template <typename DriverType>
struct HIPParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                   HIPLaunchMechanism::LocalMemory> {
  static constexpr auto default_launchbounds() { return true; }

  static auto get_kernel_func() {
    return HIPParallelLaunchKernelFunc<
        DriverType, Kokkos::LaunchBounds<HIPTraits::MaxThreadsPerBlock, 1>,
        HIPLaunchMechanism::LocalMemory>::get_kernel_func();
  }

  static unsigned int get_scratch_size() {
    return get_hip_func_attributes().localSizeBytes;
  }

  static hipFuncAttributes get_hip_func_attributes() {
    static hipFuncAttributes attr = []() {
      hipFuncAttributes attr;
      HIP_SAFE_CALL(hipFuncGetAttributes(
          &attr, reinterpret_cast<void const *>(get_kernel_func())));
      return attr;
    }();
    return attr;
  }
};

template <typename DriverType, typename LaunchBounds,
          HIPLaunchMechanism LaunchMechanism>
struct HIPParallelLaunchKernelInvoker;

template <typename DriverType, typename LaunchBounds>
struct HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                      HIPLaunchMechanism::LocalMemory>
    : HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  HIPLaunchMechanism::LocalMemory> {
  using base_t = HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                             HIPLaunchMechanism::LocalMemory>;

  static void invoke_kernel(DriverType const *driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            HIPInternal const *hip_instance) {
    (base_t::get_kernel_func())<<<grid, block, shmem, hip_instance->m_stream>>>(
        driver);
  }
};

template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          HIPLaunchMechanism LaunchMechanism = HIPLaunchMechanism::LocalMemory>
struct HIPParallelLaunch;

template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunch<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    HIPLaunchMechanism::LocalMemory>
    : HIPParallelLaunchKernelInvoker<
          DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
          HIPLaunchMechanism::LocalMemory> {
  using base_t = HIPParallelLaunchKernelInvoker<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      HIPLaunchMechanism::LocalMemory>;

  HIPParallelLaunch(const DriverType &driver, const dim3 &grid,
                    const dim3 &block, const int shmem,
                    const HIPInternal *hip_instance,
                    const bool /*prefer_shmem*/) {
    if ((grid.x != 0) && ((block.x * block.y * block.z) != 0)) {
      if (hip_instance->m_maxShmemPerBlock < shmem) {
        Kokkos::Impl::throw_runtime_exception(
            "HIPParallelLaunch FAILED: shared memory request is too large");
      }

      KOKKOS_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE();

      // Invoke the driver function on the device
      DriverType *d_driver = reinterpret_cast<DriverType *>(
          hip_instance->get_next_driver(sizeof(DriverType)));
      std::memcpy((void *)d_driver, (void *)&driver, sizeof(DriverType));
      base_t::invoke_kernel(d_driver, grid, block, shmem, hip_instance);

#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
      HIP_SAFE_CALL(hipGetLastError());
      hip_instance->fence();
#endif
    }
  }
};

template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          HIPLaunchMechanism LaunchMechanism = HIPLaunchMechanism::LocalMemory>
unsigned get_preferred_blocksize_impl() {
  // FIXME_HIP - could be if constexpr for c++17
  if (!HIPParallelLaunch<DriverType, LaunchBounds,
                         LaunchMechanism>::default_launchbounds()) {
    // use the user specified value
    return LaunchBounds::maxTperB;
  } else {
    if (static_cast<bool>(
            HIPParallelLaunch<DriverType, LaunchBounds,
                              LaunchMechanism>::get_scratch_size())) {
      return HIPTraits::ConservativeThreadsPerBlock;
    }
    return HIPTraits::MaxThreadsPerBlock;
  }
}

// FIXME_HIP - entire function could be constexpr for c++17
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          HIPLaunchMechanism LaunchMechanism = HIPLaunchMechanism::LocalMemory>
unsigned get_max_blocksize_impl() {
  // FIXME_HIP - could be if constexpr for c++17
  if (!HIPParallelLaunch<DriverType, LaunchBounds,
                         LaunchMechanism>::default_launchbounds()) {
    // use the user specified value
    return LaunchBounds::maxTperB;
  } else {
    // we can always fit 1024 threads blocks if we only care about registers
    // ... and don't mind spilling
    return HIPTraits::MaxThreadsPerBlock;
  }
}

// convenience method to select and return the proper function attributes
// for a kernel, given the launch bounds et al.
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          BlockType BlockSize                = BlockType::Max,
          HIPLaunchMechanism LaunchMechanism = HIPLaunchMechanism::LocalMemory>
hipFuncAttributes get_hip_func_attributes_impl() {
  // FIXME_HIP - could be if constexpr for c++17
  if (!HIPParallelLaunch<DriverType, LaunchBounds,
                         LaunchMechanism>::default_launchbounds()) {
    // for user defined, we *always* honor the request
    return HIPParallelLaunch<DriverType, LaunchBounds,
                             LaunchMechanism>::get_hip_func_attributes();
  } else {
    // FIXME_HIP - could be if constexpr for c++17
    if (BlockSize == BlockType::Max) {
      return HIPParallelLaunch<
          DriverType, Kokkos::LaunchBounds<HIPTraits::MaxThreadsPerBlock, 1>,
          LaunchMechanism>::get_hip_func_attributes();
    } else {
      const int blocksize =
          get_preferred_blocksize_impl<DriverType, LaunchBounds,
                                       LaunchMechanism>();
      if (blocksize == HIPTraits::MaxThreadsPerBlock) {
        return HIPParallelLaunch<
            DriverType, Kokkos::LaunchBounds<HIPTraits::MaxThreadsPerBlock, 1>,
            LaunchMechanism>::get_hip_func_attributes();
      } else {
        return HIPParallelLaunch<
            DriverType,
            Kokkos::LaunchBounds<HIPTraits::ConservativeThreadsPerBlock, 1>,
            LaunchMechanism>::get_hip_func_attributes();
      }
    }
  }
}

// convenience method to launch the correct kernel given the launch bounds et
// al.
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          HIPLaunchMechanism LaunchMechanism = HIPLaunchMechanism::LocalMemory>
void hip_parallel_launch(const DriverType &driver, const dim3 &grid,
                         const dim3 &block, const int shmem,
                         const HIPInternal *hip_instance,
                         const bool prefer_shmem) {
  // FIXME_HIP - could be if constexpr for c++17
  if (!HIPParallelLaunch<DriverType, LaunchBounds,
                         LaunchMechanism>::default_launchbounds()) {
    // for user defined, we *always* honor the request
    HIPParallelLaunch<DriverType, LaunchBounds, LaunchMechanism>(
        driver, grid, block, shmem, hip_instance, prefer_shmem);
  } else {
    // we can do what we like
    const unsigned flat_block_size = block.x * block.y * block.z;
    if (flat_block_size <= HIPTraits::ConservativeThreadsPerBlock) {
      // we have to use the large blocksize
      HIPParallelLaunch<
          DriverType,
          Kokkos::LaunchBounds<HIPTraits::ConservativeThreadsPerBlock, 1>,
          LaunchMechanism>(driver, grid, block, shmem, hip_instance,
                           prefer_shmem);
    } else {
      HIPParallelLaunch<DriverType,
                        Kokkos::LaunchBounds<HIPTraits::MaxThreadsPerBlock, 1>,
                        LaunchMechanism>(driver, grid, block, shmem,
                                         hip_instance, prefer_shmem);
    }
  }
}
}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif

#endif
