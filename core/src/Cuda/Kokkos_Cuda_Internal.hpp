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

#ifndef KOKKOS_CUDA_INTERNAL_HPP
#define KOKKOS_CUDA_INTERNAL_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include<iostream>
#include <Cuda/Kokkos_Cuda_Error.hpp>

namespace Kokkos { namespace Impl {

template<class DriverType, class LaunchBounds, bool Large>
struct CudaGetMaxBlockSize;

template<class DriverType, class LaunchBounds>
int cuda_get_max_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
  return CudaGetMaxBlockSize<DriverType,LaunchBounds,(CudaTraits::ConstantMemoryUseThreshold < sizeof(DriverType))>::get_block_size(f,vector_length, shmem_extra_block,shmem_extra_thread);
}


template<class DriverType>
struct CudaGetMaxBlockSize<DriverType,Kokkos::LaunchBounds<>,true> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int numBlocks;
    int blockSize=1024;
    int sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                    FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        cuda_parallel_launch_constant_memory<DriverType>,
        blockSize,
        sharedmem);

    if(numBlocks>0) return blockSize;
    while (blockSize>32 && numBlocks==0) {
      blockSize/=2;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          cuda_parallel_launch_constant_memory<DriverType>,
          blockSize,
          sharedmem);
    }
    int blockSizeUpperBound = blockSize*2;
    while (blockSize<blockSizeUpperBound && numBlocks>0) {
      blockSize+=32;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            cuda_parallel_launch_constant_memory<DriverType>,
            blockSize,
            sharedmem);
    }
    return blockSize - 32;
  }
};

template<class DriverType>
struct CudaGetMaxBlockSize<DriverType,Kokkos::LaunchBounds<>,false> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int numBlocks;

    unsigned int blockSize=1024;
    unsigned int sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                    FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        cuda_parallel_launch_local_memory<DriverType>,
        blockSize,
        sharedmem);

    if(numBlocks>0) return blockSize;
    while (blockSize>32 && numBlocks==0) {
      blockSize/=2;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          cuda_parallel_launch_local_memory<DriverType>,
          blockSize,
          sharedmem);
    }
    unsigned int blockSizeUpperBound = blockSize*2;
    while (blockSize<blockSizeUpperBound && numBlocks>0) {
      blockSize+=32;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            cuda_parallel_launch_local_memory<DriverType>,
            blockSize,
            sharedmem);
    }
    return blockSize - 32;
  }
};

template<class DriverType, unsigned int MaxThreadsPerBlock, unsigned int MinBlocksPerSM>
struct CudaGetMaxBlockSize<DriverType,Kokkos::LaunchBounds<MaxThreadsPerBlock,MinBlocksPerSM>,true> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int numBlocks = 0, oldNumBlocks = 0;
    unsigned int blockSize=MaxThreadsPerBlock;
    unsigned int sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                    FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        cuda_parallel_launch_constant_memory<DriverType,MaxThreadsPerBlock,MinBlocksPerSM>,
        blockSize,
        sharedmem);

    if(static_cast<unsigned int>(numBlocks)>=MinBlocksPerSM) return blockSize;

    while (blockSize>32 && static_cast<unsigned int>(numBlocks)<MinBlocksPerSM) {
      blockSize/=2;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          cuda_parallel_launch_constant_memory<DriverType>,
          blockSize,
          sharedmem);
    }
    unsigned int blockSizeUpperBound = (blockSize*2<MaxThreadsPerBlock?blockSize*2:MaxThreadsPerBlock);
    while (blockSize<blockSizeUpperBound && static_cast<unsigned int>(numBlocks)>MinBlocksPerSM) {
      blockSize+=32;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
      oldNumBlocks = numBlocks;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            cuda_parallel_launch_constant_memory<DriverType>,
            blockSize,
            sharedmem);
    }
    if(static_cast<unsigned int>(oldNumBlocks)>=MinBlocksPerSM) return blockSize - 32;
    return -1;
  }
};

template<class DriverType, unsigned int MaxThreadsPerBlock, unsigned int MinBlocksPerSM>
struct CudaGetMaxBlockSize<DriverType,Kokkos::LaunchBounds<MaxThreadsPerBlock,MinBlocksPerSM>,false> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int numBlocks = 0, oldNumBlocks = 0;
    unsigned int blockSize=MaxThreadsPerBlock;
    int sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                    FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        cuda_parallel_launch_local_memory<DriverType,MaxThreadsPerBlock,MinBlocksPerSM>,
        blockSize,
        sharedmem);
    if(static_cast<unsigned int>(numBlocks)>=MinBlocksPerSM) return blockSize;

    while (blockSize>32 && static_cast<unsigned int>(numBlocks)<MinBlocksPerSM) {
      blockSize/=2;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocks,
          cuda_parallel_launch_local_memory<DriverType>,
          blockSize,
          sharedmem);
    }
    unsigned int blockSizeUpperBound = (blockSize*2<MaxThreadsPerBlock?blockSize*2:MaxThreadsPerBlock);
    while (blockSize<blockSizeUpperBound && static_cast<unsigned int>(numBlocks)>=MinBlocksPerSM) {
      blockSize+=32;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
      oldNumBlocks = numBlocks;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            cuda_parallel_launch_local_memory<DriverType>,
            blockSize,
            sharedmem);
    }
    if(static_cast<unsigned int>(oldNumBlocks)>=MinBlocksPerSM) return blockSize - 32;
    return -1;
  }
};


template<class DriverType, class LaunchBounds, bool Large>
struct CudaGetOptBlockSize;

template<class DriverType, class LaunchBounds>
int cuda_get_opt_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
  return CudaGetOptBlockSize<DriverType,LaunchBounds,(CudaTraits::ConstantMemoryUseThreshold < sizeof(DriverType))>::get_block_size(f,vector_length,shmem_extra_block,shmem_extra_thread);
}

template<class DriverType>
struct CudaGetOptBlockSize<DriverType,Kokkos::LaunchBounds<>,true> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int blockSize=16;
    int numBlocks;
    int sharedmem;
    int maxOccupancy=0;
    int bestBlockSize=0;

    while(blockSize<1024) {
      blockSize*=2;

      //calculate the occupancy with that optBlockSize and check whether its larger than the largest one found so far
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &numBlocks,
              cuda_parallel_launch_constant_memory<DriverType>,
              blockSize,
              sharedmem);
      if(maxOccupancy < numBlocks*blockSize) {
         maxOccupancy = numBlocks*blockSize;
         bestBlockSize = blockSize;
      }
    }
    return bestBlockSize;
  }
};

template<class DriverType>
struct CudaGetOptBlockSize<DriverType,Kokkos::LaunchBounds<>,false> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int blockSize=16;
    int numBlocks;
    int sharedmem;
    int maxOccupancy=0;
    int bestBlockSize=0;

    while(blockSize<1024) {
      blockSize*=2;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &numBlocks,
              cuda_parallel_launch_local_memory<DriverType>,
              blockSize,
              sharedmem);

      if(maxOccupancy < numBlocks*blockSize) {
        maxOccupancy = numBlocks*blockSize;
        bestBlockSize = blockSize;
      }
    }
    return bestBlockSize;
  }
};

template<class DriverType, unsigned int MaxThreadsPerBlock, unsigned int MinBlocksPerSM>
struct CudaGetOptBlockSize<DriverType,Kokkos::LaunchBounds< MaxThreadsPerBlock, MinBlocksPerSM >,true> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int blockSize=16;
    int numBlocks;
    int sharedmem;
    int maxOccupancy=0;
    int bestBlockSize=0;
    int max_threads_per_block = std::min(MaxThreadsPerBlock,cuda_internal_maximum_warp_count()*CudaTraits::WarpSize);

    while(blockSize < max_threads_per_block ) {
      blockSize*=2;

      //calculate the occupancy with that optBlockSize and check whether its larger than the largest one found so far
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &numBlocks,
              cuda_parallel_launch_constant_memory<DriverType,MaxThreadsPerBlock,MinBlocksPerSM>,
              blockSize,
              sharedmem);
      if(numBlocks >= int(MinBlocksPerSM) && blockSize<=int(MaxThreadsPerBlock)) {
        if(maxOccupancy < numBlocks*blockSize) {
           maxOccupancy = numBlocks*blockSize;
           bestBlockSize = blockSize;
        }
      }
    }
    if(maxOccupancy > 0)
      return bestBlockSize;
    return -1;
  }
};

template<class DriverType, unsigned int MaxThreadsPerBlock, unsigned int MinBlocksPerSM>
struct CudaGetOptBlockSize<DriverType,Kokkos::LaunchBounds< MaxThreadsPerBlock, MinBlocksPerSM >,false> {
  static int get_block_size(const typename DriverType::functor_type & f, const size_t vector_length,
                            const size_t shmem_extra_block, const size_t shmem_extra_thread) {
    int blockSize=16;
    int numBlocks;
    int sharedmem;
    int maxOccupancy=0;
    int bestBlockSize=0;
    int max_threads_per_block = std::min(MaxThreadsPerBlock,cuda_internal_maximum_warp_count()*CudaTraits::WarpSize);

    while(blockSize < max_threads_per_block ) {
      blockSize*=2;
      sharedmem = shmem_extra_block + shmem_extra_thread*(blockSize/vector_length) +
                  FunctorTeamShmemSize< typename DriverType::functor_type  >::value( f , blockSize/vector_length );

      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &numBlocks,
              cuda_parallel_launch_local_memory<DriverType,MaxThreadsPerBlock,MinBlocksPerSM>,
              blockSize,
              sharedmem);
      if(numBlocks >= int(MinBlocksPerSM) && blockSize<=int(MaxThreadsPerBlock)) {
        if(maxOccupancy < numBlocks*blockSize) {
          maxOccupancy = numBlocks*blockSize;
          bestBlockSize = blockSize;
        }
      }
    }
    if(maxOccupancy > 0)
      return bestBlockSize;
    return -1;
  }
};

}} // namespace Kokkos::Impl

#endif // KOKKOS_ENABLE_CUDA
#endif /* #ifndef KOKKOS_CUDA_INTERNAL_HPP */

