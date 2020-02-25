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

#ifndef KOKKOS_HIP_BLOCKSIZE_DEDUCTION_HPP
#define KOKKOS_HIP_BLOCKSIZE_DEDUCTION_HPP

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_HIP) && defined(__HIPCC__)

#include <HIP/Kokkos_HIP_Instance.hpp>
#include <HIP/Kokkos_HIP_KernelLaunch.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {
template <typename DriverType, typename LaunchBounds, bool Large>
struct HIPGetMaxBlockSize;

// FIXME_HIP In CUDA we have a variant which gets the functor attributes and
// then uses information from that to figure out what to do instead of calling
// repeatedly the occupancy function from the runtime.

template <typename DriverType, typename LaunchBounds>
int hip_get_max_block_size(typename DriverType::functor_type const &f,
                           size_t const vector_length,
                           size_t const shmem_extra_block,
                           size_t const shmem_extra_thread) {
  return HIPGetMaxBlockSize<DriverType, LaunchBounds, true>::get_block_size(
      f, vector_length, shmem_extra_block, shmem_extra_thread);
}

template <typename DriverType>
struct HIPGetMaxBlockSize<DriverType, Kokkos::LaunchBounds<>, true> {
  static int get_block_size(typename DriverType::functor_type const &f,
                            size_t const vector_length,
                            size_t const shmem_extra_block,
                            size_t const shmem_extra_thread) {
    unsigned int numBlocks = 0;
    int blockSize          = 1024;
    int sharedmem =
        shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
        ::Kokkos::Impl::FunctorTeamShmemSize<
            typename DriverType::functor_type>::value(f, blockSize /
                                                             vector_length);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, hip_parallel_launch_constant_memory<DriverType>, blockSize,
        sharedmem);

    if (numBlocks > 0) return blockSize;
    while (blockSize > HIPTraits::WarpSize && numBlocks == 0) {
      blockSize /= 2;
      sharedmem =
          shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
          ::Kokkos::Impl::FunctorTeamShmemSize<
              typename DriverType::functor_type>::value(f, blockSize /
                                                               vector_length);

      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, hip_parallel_launch_constant_memory<DriverType>,
          blockSize, sharedmem);
    }
    int blockSizeUpperBound = blockSize * 2;
    while (blockSize < blockSizeUpperBound && numBlocks > 0) {
      blockSize += HIPTraits::WarpSize;
      sharedmem =
          shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
          ::Kokkos::Impl::FunctorTeamShmemSize<
              typename DriverType::functor_type>::value(f, blockSize /
                                                               vector_length);

      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks, hip_parallel_launch_constant_memory<DriverType>,
          blockSize, sharedmem);
    }
    return blockSize - HIPTraits::WarpSize;
  }
};
}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif

#endif
