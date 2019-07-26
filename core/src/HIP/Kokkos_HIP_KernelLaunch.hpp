/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include <HIP/Kokkos_HIP_Error.hpp>
#include <HIP/Kokkos_HIP_Instance.hpp>
#include <Kokkos_HIP.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class DriverType>
__global__ static void hip_parallel_launch_local_memory(
    const DriverType driver) {
  driver();
}

template <class DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ __launch_bounds__(
    maxTperB,
    minBperSM) static void hip_parallel_launch_local_memory(const DriverType
                                                                driver) {
  driver();
}

enum class HIPLaunchMechanism : unsigned {
  Default        = 0,
  ConstantMemory = 1,
  GlobalMemory   = 2,
  LocalMemory    = 4
};

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

template <class DriverType, class LaunchBounds = Kokkos::LaunchBounds<>,
          HIPLaunchMechanism LaunchMechanism = HIPLaunchMechanism::LocalMemory>
struct HIPParallelLaunch;

template <class DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunch<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    HIPLaunchMechanism::LocalMemory> {
  // static_assert(sizeof(DriverType)<HIPTraits::KernelArgumentLimit,"Kokkos
  // Error: Requested HIPLaunchLocalMemory with a Functor larger than 4096
  // bytes.");
  inline HIPParallelLaunch(const DriverType &driver, const dim3 &grid,
                           const dim3 &block, const int shmem,
                           const HIPInternal *hip_instance,
                           const bool prefer_shmem) {
    if ((grid.x != 0) && ((block.x * block.y * block.z) != 0)) {
      /*
            if ( hip_instance->m_maxShmemPerBlock < shmem ) {
              Kokkos::Impl::throw_runtime_exception(
         std::string("HIPParallelLaunch FAILED: shared memory request is too
         large") );
            }
            #ifndef KOKKOS_ARCH_KEPLER
            // On Kepler the L1 has no benefit since it doesn't cache reads
            else {
              CUDA_SAFE_CALL(
                hipFuncSetCacheConfig
                  ( hip_parallel_launch_local_memory
                      < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
                  , ( prefer_shmem ? hipFuncCachePreferShared :
         hipFuncCachePreferL1 ) ) );
            }
            #endif

            KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();*/

      // Invoke the driver function on the device
      printf("%i %i %i | %i %i %i | %i\n", grid.x, grid.y, grid.z, block.x,
             block.y, block.z, shmem);
      printf("Pre Launch Error: %s\n", hipGetErrorName(hipGetLastError()));

      hipLaunchKernelGGL(
          hip_parallel_launch_local_memory<DriverType, MaxThreadsPerBlock,
                                           MinBlocksPerSM>,
          grid, block, shmem, hip_instance->m_stream, driver);

      Kokkos::Experimental::HIP().fence();
      printf("Post Launch Error: %s\n", hipGetErrorName(hipGetLastError()));
#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
      // CUDA_SAFE_CALL( hipGetLastError() );
      Kokkos::HIP().fence();
#endif
    }
  }
  /*
    static hipFuncAttributes get_hip_func_attributes() {
      hipFuncAttributes attr;
      hipFuncGetAttributes(&attr,hip_parallel_launch_local_memory
              < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >);
      return attr;
    }*/
};

template <class DriverType>
struct HIPParallelLaunch<DriverType, Kokkos::LaunchBounds<0, 0>,
                         HIPLaunchMechanism::LocalMemory> {
  // static_assert(sizeof(DriverType)<HIPTraits::KernelArgumentLimit,"Kokkos
  // Error: Requested HIPLaunchLocalMemory with a Functor larger than 4096
  // bytes.");
  inline HIPParallelLaunch(const DriverType &driver, const dim3 &grid,
                           const dim3 &block, const int shmem,
                           const HIPInternal *hip_instance,
                           const bool prefer_shmem) {
    if ((grid.x != 0) && ((block.x * block.y * block.z) != 0)) {
      /**
            if ( hip_instance->m_maxShmemPerBlock < shmem ) {
              Kokkos::Impl::throw_runtime_exception(
         std::string("HIPParallelLaunch FAILED: shared memory request is too
         large") );
            }

            KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();
      */
      // Invoke the driver function on the device
      hipLaunchKernelGGL(hip_parallel_launch_local_memory<DriverType>, grid,
                         block, shmem, hip_instance->m_stream, driver);

      Kokkos::Experimental::HIP().fence();
#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
      HIP_SAFE_CALL(hipGetLastError());
      Kokkos::HIP().fence();
#endif
    }
  }
  /*
    static hipFuncAttributes get_hip_func_attributes() {
      hipFuncAttributes attr;
      hipFuncGetAttributes(&attr,hip_parallel_launch_local_memory
              < DriverType >);
      return attr;
    }*/
};
}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
