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

#ifndef KOKKOS_CUDAEXEC_HPP
#define KOKKOS_CUDAEXEC_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <string>
#include <cstdint>
#include <Kokkos_Parallel.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Cuda/Kokkos_Cuda_abort.hpp>
#include <Cuda/Kokkos_Cuda_Error.hpp>
#include <Cuda/Kokkos_Cuda_Locks.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct CudaTraits {
  enum { WarpSize       = 32      /* 0x0020 */ };
  enum { WarpIndexMask  = 0x001f  /* Mask for warpindex */ };
  enum { WarpIndexShift = 5       /* WarpSize == 1 << WarpShift */ };

  enum { SharedMemoryBanks    = 32      /* Compute device 2.0 */ };
  enum { SharedMemoryCapacity = 0x0C000 /* 48k shared / 16k L1 Cache */ };
  enum { SharedMemoryUsage    = 0x04000 /* 16k shared / 48k L1 Cache */ };

  enum { UpperBoundGridCount    = 65535 /* Hard upper bound */ };
  enum { ConstantMemoryCapacity = 0x010000 /* 64k bytes */ };
  enum { ConstantMemoryUsage    = 0x008000 /* 32k bytes */ };
  enum { ConstantMemoryCache    = 0x002000 /*  8k bytes */ };

  typedef unsigned long
    ConstantGlobalBufferType[ ConstantMemoryUsage / sizeof(unsigned long) ];

  enum { ConstantMemoryUseThreshold = 0x000200 /* 512 bytes */ };

  KOKKOS_INLINE_FUNCTION static
  CudaSpace::size_type warp_count( CudaSpace::size_type i )
    { return ( i + WarpIndexMask ) >> WarpIndexShift ; }

  KOKKOS_INLINE_FUNCTION static
  CudaSpace::size_type warp_align( CudaSpace::size_type i )
    {
      enum { Mask = ~CudaSpace::size_type( WarpIndexMask ) };
      return ( i + WarpIndexMask ) & Mask ;
    }
};

//----------------------------------------------------------------------------

CudaSpace::size_type cuda_internal_multiprocessor_count();
CudaSpace::size_type cuda_internal_maximum_warp_count();
CudaSpace::size_type cuda_internal_maximum_grid_count();
CudaSpace::size_type cuda_internal_maximum_shared_words();

CudaSpace::size_type cuda_internal_maximum_concurrent_block_count();

CudaSpace::size_type * cuda_internal_scratch_flags( const CudaSpace::size_type size );
CudaSpace::size_type * cuda_internal_scratch_space( const CudaSpace::size_type size );
CudaSpace::size_type * cuda_internal_scratch_unified( const CudaSpace::size_type size );

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#if defined( __CUDACC__ )

/** \brief  Access to constant memory on the device */
#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE

__device__ __constant__
extern unsigned long kokkos_impl_cuda_constant_memory_buffer[] ;

#else

__device__ __constant__
unsigned long kokkos_impl_cuda_constant_memory_buffer[ Kokkos::Impl::CudaTraits::ConstantMemoryUsage / sizeof(unsigned long) ] ;

#endif

namespace Kokkos {
namespace Impl {
  void* cuda_resize_scratch_space(std::int64_t bytes, bool force_shrink = false);
}
}

template< typename T >
inline
__device__
T * kokkos_impl_cuda_shared_memory()
{ extern __shared__ Kokkos::CudaSpace::size_type sh[]; return (T*) sh ; }

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
// See section B.17 of Cuda C Programming Guide Version 3.2
// for discussion of
//   __launch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)
// function qualifier which could be used to improve performance.
//----------------------------------------------------------------------------
// Maximize L1 cache and minimize shared memory:
//   cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferL1 );
// For 2.0 capability: 48 KB L1 and 16 KB shared
//----------------------------------------------------------------------------

template< class DriverType>
__global__
static void cuda_parallel_launch_constant_memory()
{
  const DriverType & driver =
    *((const DriverType *) kokkos_impl_cuda_constant_memory_buffer );

  driver();
}

template< class DriverType, unsigned int maxTperB, unsigned int minBperSM >
__global__
__launch_bounds__(maxTperB, minBperSM)
static void cuda_parallel_launch_constant_memory()
{
  const DriverType & driver =
    *((const DriverType *) kokkos_impl_cuda_constant_memory_buffer );

  driver();
}

template< class DriverType>
__global__
static void cuda_parallel_launch_local_memory( const DriverType driver )
{
  driver();
}

template< class DriverType, unsigned int maxTperB, unsigned int minBperSM >
__global__
__launch_bounds__(maxTperB, minBperSM)
static void cuda_parallel_launch_local_memory( const DriverType driver )
{
  driver();
}

template < class DriverType
         , class LaunchBounds = Kokkos::LaunchBounds<>
         , bool Large = ( CudaTraits::ConstantMemoryUseThreshold < sizeof(DriverType) ) >
struct CudaParallelLaunch ;

template < class DriverType
         , unsigned int MaxThreadsPerBlock
         , unsigned int MinBlocksPerSM >
struct CudaParallelLaunch< DriverType
                         , Kokkos::LaunchBounds< MaxThreadsPerBlock 
                                               , MinBlocksPerSM >
                         , true >
{
  inline
  CudaParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const cudaStream_t stream = 0 )
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {

      if ( sizeof( Kokkos::Impl::CudaTraits::ConstantGlobalBufferType ) <
           sizeof( DriverType ) ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: Functor is too large") );
      }

      // Fence before changing settings and copying closure
      Kokkos::Cuda::fence();

      if ( CudaTraits::SharedMemoryCapacity < shmem ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: shared memory request is too large") );
      }
      #ifndef KOKKOS_ARCH_KEPLER
      // On Kepler the L1 has no benefit since it doesn't cache reads
      else {
        CUDA_SAFE_CALL(
          cudaFuncSetCacheConfig
            ( cuda_parallel_launch_constant_memory
                < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
            , ( shmem ? cudaFuncCachePreferShared : cudaFuncCachePreferL1 )
            ) );
      }
      #endif

      // Copy functor to constant memory on the device
      cudaMemcpyToSymbol(
        kokkos_impl_cuda_constant_memory_buffer, &driver, sizeof(DriverType) );

      KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();

      // Invoke the driver function on the device
      cuda_parallel_launch_constant_memory
        < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
          <<< grid , block , shmem , stream >>>();

#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
      CUDA_SAFE_CALL( cudaGetLastError() );
      Kokkos::Cuda::fence();
#endif
    }
  }
};

template < class DriverType >
struct CudaParallelLaunch< DriverType
                         , Kokkos::LaunchBounds<>
                         , true >
{
  inline
  CudaParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const cudaStream_t stream = 0 )
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {

      if ( sizeof( Kokkos::Impl::CudaTraits::ConstantGlobalBufferType ) <
           sizeof( DriverType ) ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: Functor is too large") );
      }

      // Fence before changing settings and copying closure
      Kokkos::Cuda::fence();

      if ( CudaTraits::SharedMemoryCapacity < shmem ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: shared memory request is too large") );
      }
      #ifndef KOKKOS_ARCH_KEPLER
      // On Kepler the L1 has no benefit since it doesn't cache reads
      else {
        CUDA_SAFE_CALL(
          cudaFuncSetCacheConfig
            ( cuda_parallel_launch_constant_memory< DriverType >
            , ( shmem ? cudaFuncCachePreferShared : cudaFuncCachePreferL1 )
            ) );
      }
      #endif

      // Copy functor to constant memory on the device
      cudaMemcpyToSymbol(
        kokkos_impl_cuda_constant_memory_buffer, &driver, sizeof(DriverType) );

      KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();

      // Invoke the driver function on the device
      cuda_parallel_launch_constant_memory< DriverType >
          <<< grid , block , shmem , stream >>>();

#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
      CUDA_SAFE_CALL( cudaGetLastError() );
      Kokkos::Cuda::fence();
#endif
    }
  }
};

template < class DriverType
         , unsigned int MaxThreadsPerBlock
         , unsigned int MinBlocksPerSM >
struct CudaParallelLaunch< DriverType
                         , Kokkos::LaunchBounds< MaxThreadsPerBlock 
                                               , MinBlocksPerSM >
                         , false >
{
  inline
  CudaParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const cudaStream_t stream = 0 )
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {

      if ( sizeof( Kokkos::Impl::CudaTraits::ConstantGlobalBufferType ) <
           sizeof( DriverType ) ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: Functor is too large") );
      }

      if ( CudaTraits::SharedMemoryCapacity < shmem ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: shared memory request is too large") );
      }
      #ifndef KOKKOS_ARCH_KEPLER
      // On Kepler the L1 has no benefit since it doesn't cache reads
      else {
        CUDA_SAFE_CALL(
          cudaFuncSetCacheConfig
            ( cuda_parallel_launch_local_memory
                < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
            , ( shmem ? cudaFuncCachePreferShared : cudaFuncCachePreferL1 )
            ) );
      }
      #endif

      KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();

      // Invoke the driver function on the device
      cuda_parallel_launch_local_memory
        < DriverType, MaxThreadsPerBlock, MinBlocksPerSM >
          <<< grid , block , shmem , stream >>>( driver );

#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
      CUDA_SAFE_CALL( cudaGetLastError() );
      Kokkos::Cuda::fence();
#endif
    }
  }
};

template < class DriverType >
struct CudaParallelLaunch< DriverType
                         , Kokkos::LaunchBounds<>
                         , false >
{
  inline
  CudaParallelLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
                    , const int          shmem
                    , const cudaStream_t stream = 0 )
  {
    if ( (grid.x != 0) && ( ( block.x * block.y * block.z ) != 0 ) ) {

      if ( sizeof( Kokkos::Impl::CudaTraits::ConstantGlobalBufferType ) <
           sizeof( DriverType ) ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: Functor is too large") );
      }

      if ( CudaTraits::SharedMemoryCapacity < shmem ) {
        Kokkos::Impl::throw_runtime_exception( std::string("CudaParallelLaunch FAILED: shared memory request is too large") );
      }
      #ifndef KOKKOS_ARCH_KEPLER
      // On Kepler the L1 has no benefit since it doesn't cache reads
      else {
        CUDA_SAFE_CALL(
          cudaFuncSetCacheConfig
            ( cuda_parallel_launch_local_memory< DriverType >
            , ( shmem ? cudaFuncCachePreferShared : cudaFuncCachePreferL1 )
            ) );
      }
      #endif

      KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE();

      // Invoke the driver function on the device
      cuda_parallel_launch_local_memory< DriverType >
          <<< grid , block , shmem , stream >>>( driver );

#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
      CUDA_SAFE_CALL( cudaGetLastError() );
      Kokkos::Cuda::fence();
#endif
    }
  }
};

//----------------------------------------------------------------------------

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* defined( __CUDACC__ ) */
#endif /* defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #ifndef KOKKOS_CUDAEXEC_HPP */

