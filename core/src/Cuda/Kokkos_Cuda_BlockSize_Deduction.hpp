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

#ifndef KOKKOS_CUDA_INTERNAL_HPP
#define KOKKOS_CUDA_INTERNAL_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <iostream>
#include <Cuda/Kokkos_Cuda_Error.hpp>

namespace Kokkos {
namespace Impl {

template <class DriverType, class LaunchBounds, bool Large>
struct CudaGetMaxBlockSize;

template <class FunctorType, class LaunchBounds>
int cuda_get_max_block_size(const CudaInternal* cuda_instance,
                            const cudaFuncAttributes& attr,
                            const FunctorType& f, const size_t vector_length,
                            const size_t shmem_block,
                            const size_t shmem_thread) {
  const int min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : LaunchBounds::minBperSM;
  const int max_threads_per_block = LaunchBounds::maxTperB == 0
                                        ? cuda_instance->m_maxThreadsPerBlock
                                        : LaunchBounds::maxTperB;

  const int regs_per_thread     = attr.numRegs;
  const int regs_per_sm         = cuda_instance->m_regsPerSM;
  const int shmem_per_sm        = cuda_instance->m_shmemPerSM;
  const int max_shmem_per_block = cuda_instance->m_maxShmemPerBlock;
  const int max_blocks_per_sm   = cuda_instance->m_maxBlocksPerSM;
  const int max_threads_per_sm  = cuda_instance->m_maxThreadsPerSM;

  int block_size = std::min(attr.maxThreadsPerBlock, max_threads_per_block);

  int functor_shmem =
      FunctorTeamShmemSize<FunctorType>::value(f, block_size / vector_length);
  int total_shmem = shmem_block + shmem_thread * (block_size / vector_length) +
                    functor_shmem + attr.sharedSizeBytes;
  int max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);
  int max_blocks_shmem =
      (total_shmem < max_shmem_per_block)
          ? (total_shmem > 0 ? shmem_per_sm / total_shmem : max_blocks_regs)
          : 0;
  int blocks_per_sm  = std::min(max_blocks_regs, max_blocks_shmem);
  int threads_per_sm = blocks_per_sm * block_size;
  if (threads_per_sm > max_threads_per_sm) {
    blocks_per_sm  = max_threads_per_sm / block_size;
    threads_per_sm = blocks_per_sm * block_size;
  }
  int opt_block_size = (blocks_per_sm >= min_blocks_per_sm) ? block_size : 0;
  int opt_threads_per_sm = threads_per_sm;
  // printf("BlockSizeMax: %i Shmem: %i %i %i %i Regs: %i %i Blocks: %i %i
  // Achieved: %i %i Opt: %i %i\n",block_size,
  //   shmem_per_sm,max_shmem_per_block,functor_shmem,total_shmem,
  //   regs_per_sm,regs_per_thread,max_blocks_shmem,max_blocks_regs,blocks_per_sm,threads_per_sm,opt_block_size,opt_threads_per_sm);
  block_size -= 32;
  while ((blocks_per_sm == 0) && (block_size >= 32)) {
    functor_shmem =
        FunctorTeamShmemSize<FunctorType>::value(f, block_size / vector_length);
    total_shmem = shmem_block + shmem_thread * (block_size / vector_length) +
                  functor_shmem + attr.sharedSizeBytes;
    max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);
    max_blocks_shmem =
        (total_shmem < max_shmem_per_block)
            ? (total_shmem > 0 ? shmem_per_sm / total_shmem : max_blocks_regs)
            : 0;
    blocks_per_sm  = std::min(max_blocks_regs, max_blocks_shmem);
    threads_per_sm = blocks_per_sm * block_size;
    if (threads_per_sm > max_threads_per_sm) {
      blocks_per_sm  = max_threads_per_sm / block_size;
      threads_per_sm = blocks_per_sm * block_size;
    }
    if ((blocks_per_sm >= min_blocks_per_sm) &&
        (blocks_per_sm <= max_blocks_per_sm)) {
      if (threads_per_sm >= opt_threads_per_sm) {
        opt_block_size     = block_size;
        opt_threads_per_sm = threads_per_sm;
      }
    }
    // printf("BlockSizeMax: %i Shmem: %i %i %i %i Regs: %i %i Blocks: %i %i
    // Achieved: %i %i Opt: %i %i\n",block_size,
    //   shmem_per_sm,max_shmem_per_block,functor_shmem,total_shmem,
    //   regs_per_sm,regs_per_thread,max_blocks_shmem,max_blocks_regs,blocks_per_sm,threads_per_sm,opt_block_size,opt_threads_per_sm);
    block_size -= 32;
  }
  return opt_block_size;
}

// Holds the kernel function for which occupancy is calculated using
// cudaOccupancyMaxActiveBlocksPerMultiprocessor
template <typename DriverType, typename LaunchBounds, bool Large>
struct CudaParallelLaunchLocalOrConstantMemory;

template <typename DriverType>
struct CudaParallelLaunchLocalOrConstantMemory<DriverType,
                                               Kokkos::LaunchBounds<>, true> {
  static constexpr auto func = cuda_parallel_launch_constant_memory<DriverType>;
};

template <typename DriverType>
struct CudaParallelLaunchLocalOrConstantMemory<DriverType,
                                               Kokkos::LaunchBounds<>, false> {
  static constexpr auto func = cuda_parallel_launch_local_memory<DriverType>;
};

// realized none of the CudaGetMaxBlockSize and CudaGetOptBlockSize
// specializations that would require them actually gets instantiated.

template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct CudaParallelLaunchLocalOrConstantMemory<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    true> {
  static constexpr auto func =
      cuda_parallel_launch_constant_memory<DriverType, MaxThreadsPerBlock,
                                           MinBlocksPerSM>;
};

template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct CudaParallelLaunchLocalOrConstantMemory<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    false> {
  static constexpr auto func =
      cuda_parallel_launch_local_memory<DriverType, MaxThreadsPerBlock,
                                        MinBlocksPerSM>;
};

template <class DriverType, bool Large>
struct CudaGetMaxBlockSize<DriverType, Kokkos::LaunchBounds<>, Large> {
  static int get_block_size(const typename DriverType::functor_type& f,
                            const size_t vector_length,
                            const size_t shmem_extra_block,
                            const size_t shmem_extra_thread) {
    int numBlocks;
    int blockSize = 1024;
    int sharedmem =
        shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
        FunctorTeamShmemSize<typename DriverType::functor_type>::value(
            f, blockSize / vector_length);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        CudaParallelLaunchLocalOrConstantMemory<
            DriverType, Kokkos::LaunchBounds<>, Large>::func,
        blockSize, sharedmem);

    if (numBlocks > 0) return blockSize;
    while (blockSize > 32 && numBlocks == 0) {
      blockSize /= 2;
      sharedmem =
          shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
          FunctorTeamShmemSize<typename DriverType::functor_type>::value(
              f, blockSize / vector_length);

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          CudaParallelLaunchLocalOrConstantMemory<
              DriverType, Kokkos::LaunchBounds<>, Large>::func,
          blockSize, sharedmem);
    }
    int blockSizeUpperBound = blockSize * 2;
    while (blockSize < blockSizeUpperBound && numBlocks > 0) {
      blockSize += 32;
      sharedmem =
          shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
          FunctorTeamShmemSize<typename DriverType::functor_type>::value(
              f, blockSize / vector_length);

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          CudaParallelLaunchLocalOrConstantMemory<
              DriverType, Kokkos::LaunchBounds<>, Large>::func,
          blockSize, sharedmem);
    }
    return blockSize - 32;
  }
};

template <class DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM, bool Large>
struct CudaGetMaxBlockSize<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    Large> {
  static int get_block_size(const typename DriverType::functor_type& f,
                            const size_t vector_length,
                            const size_t shmem_extra_block,
                            const size_t shmem_extra_thread) {
    int numBlocks = 0, oldNumBlocks = 0;
    unsigned int blockSize = MaxThreadsPerBlock;
    unsigned int sharedmem =
        shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
        FunctorTeamShmemSize<typename DriverType::functor_type>::value(
            f, blockSize / vector_length);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        CudaParallelLaunchLocalOrConstantMemory<
            DriverType,
            Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
            Large>::func,
        blockSize, sharedmem);

    if (static_cast<unsigned int>(numBlocks) >= MinBlocksPerSM)
      return blockSize;

    while (blockSize > 32 &&
           static_cast<unsigned int>(numBlocks) < MinBlocksPerSM) {
      blockSize /= 2;
      sharedmem =
          shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
          FunctorTeamShmemSize<typename DriverType::functor_type>::value(
              f, blockSize / vector_length);

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          CudaParallelLaunchLocalOrConstantMemory<
              DriverType, Kokkos::LaunchBounds</*not a typo*/>, Large>::func,
          blockSize, sharedmem);
    }
    unsigned int blockSizeUpperBound =
        (blockSize * 2 < MaxThreadsPerBlock ? blockSize * 2
                                            : MaxThreadsPerBlock);
    while (blockSize<blockSizeUpperBound&& static_cast<unsigned int>(numBlocks)>
               MinBlocksPerSM) {
      blockSize += 32;
      sharedmem =
          shmem_extra_block + shmem_extra_thread * (blockSize / vector_length) +
          FunctorTeamShmemSize<typename DriverType::functor_type>::value(
              f, blockSize / vector_length);
      oldNumBlocks = numBlocks;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          CudaParallelLaunchLocalOrConstantMemory<
              DriverType, Kokkos::LaunchBounds</*not a typo*/>, Large>::func,
          blockSize, sharedmem);
    }
    if (static_cast<unsigned int>(oldNumBlocks) >= MinBlocksPerSM)
      return blockSize - 32;
    return -1;
  }
};

template <class FunctorType, class LaunchBounds>
int cuda_get_opt_block_size(const CudaInternal* cuda_instance,
                            const cudaFuncAttributes& attr,
                            const FunctorType& f, const size_t vector_length,
                            const size_t shmem_block,
                            const size_t shmem_thread) {
  const int min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : LaunchBounds::minBperSM;
  const int max_threads_per_block = LaunchBounds::maxTperB == 0
                                        ? cuda_instance->m_maxThreadsPerBlock
                                        : LaunchBounds::maxTperB;

  const int regs_per_thread     = attr.numRegs;
  const int regs_per_sm         = cuda_instance->m_regsPerSM;
  const int shmem_per_sm        = cuda_instance->m_shmemPerSM;
  const int max_shmem_per_block = cuda_instance->m_maxShmemPerBlock;
  const int max_blocks_per_sm   = cuda_instance->m_maxBlocksPerSM;
  const int max_threads_per_sm  = cuda_instance->m_maxThreadsPerSM;

  int block_size = std::min(attr.maxThreadsPerBlock, max_threads_per_block);

  int functor_shmem =
      FunctorTeamShmemSize<FunctorType>::value(f, block_size / vector_length);
  int total_shmem = shmem_block + shmem_thread * (block_size / vector_length) +
                    functor_shmem + attr.sharedSizeBytes;
  int max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);
  int max_blocks_shmem =
      (total_shmem < max_shmem_per_block)
          ? (total_shmem > 0 ? shmem_per_sm / total_shmem : max_blocks_regs)
          : 0;
  int blocks_per_sm  = std::min(max_blocks_regs, max_blocks_shmem);
  int threads_per_sm = blocks_per_sm * block_size;
  if (threads_per_sm > max_threads_per_sm) {
    blocks_per_sm  = max_threads_per_sm / block_size;
    threads_per_sm = blocks_per_sm * block_size;
  }
  int opt_block_size = (blocks_per_sm >= min_blocks_per_sm) ? block_size : 0;
  int opt_threads_per_sm = threads_per_sm;

  block_size -= 32;
  while ((block_size >= 32)) {
    functor_shmem =
        FunctorTeamShmemSize<FunctorType>::value(f, block_size / vector_length);
    total_shmem = shmem_block + shmem_thread * (block_size / vector_length) +
                  functor_shmem + attr.sharedSizeBytes;
    max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);
    max_blocks_shmem =
        (total_shmem < max_shmem_per_block)
            ? (total_shmem > 0 ? shmem_per_sm / total_shmem : max_blocks_regs)
            : 0;
    blocks_per_sm  = std::min(max_blocks_regs, max_blocks_shmem);
    threads_per_sm = blocks_per_sm * block_size;
    if (threads_per_sm > max_threads_per_sm) {
      blocks_per_sm  = max_threads_per_sm / block_size;
      threads_per_sm = blocks_per_sm * block_size;
    }
    if ((blocks_per_sm >= min_blocks_per_sm) &&
        (blocks_per_sm <= max_blocks_per_sm)) {
      if (threads_per_sm >= opt_threads_per_sm) {
        opt_block_size     = block_size;
        opt_threads_per_sm = threads_per_sm;
      }
    }
    block_size -= 32;
  }
  return opt_block_size;
}

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_CUDA
#endif  /* #ifndef KOKKOS_CUDA_INTERNAL_HPP */
