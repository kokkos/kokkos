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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CUDA_LOCKS_HPP
#define KOKKOS_CUDA_LOCKS_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_CUDA

#include <cstdint>

#include <Cuda/Kokkos_Cuda_Error.hpp>

namespace Kokkos {
namespace Impl {

struct CudaLockArrays {
  std::int32_t* atomic;
  std::int32_t* threadid;
  std::int32_t n;
};

extern Kokkos::Impl::CudaLockArrays host_cuda_lock_arrays ;

void initialize_host_cuda_lock_arrays();
void finalize_host_cuda_lock_arrays();

} // namespace Impl
} // namespace Kokkos

#if defined( __CUDACC__ )

namespace Kokkos {
namespace Impl {

__device__ __constant__
#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
extern
#else
static
#endif
Kokkos::Impl::CudaLockArrays device_cuda_lock_arrays ;

#define CUDA_SPACE_ATOMIC_MASK 0x1FFFF

/// \brief Aquire a lock for the address
///
/// This function tries to aquire the lock for the hash value derived
/// from the provided ptr. If the lock is successfully aquired the
/// function returns true. Otherwise it returns false.
__device__ inline
bool lock_address_cuda_space(void* ptr) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & CUDA_SPACE_ATOMIC_MASK;
  return (0 == atomicCAS(&Kokkos::Impl::device_cuda_lock_arrays.atomic[offset],0,1));
}

/// \brief Release lock for the address
///
/// This function releases the lock for the hash value derived
/// from the provided ptr. This function should only be called
/// after previously successfully aquiring a lock with
/// lock_address.
__device__ inline
void unlock_address_cuda_space(void* ptr) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & CUDA_SPACE_ATOMIC_MASK;
  atomicExch( &Kokkos::Impl::device_cuda_lock_arrays.atomic[ offset ], 0);
}

} // namespace Impl
} // namespace Kokkos

/* Dan Ibanez: it is critical that this code be a macro, so that it will
   capture the right address for Kokkos::Impl::device_cuda_lock_arrays!
   putting this in an inline function will NOT do the right thing! */
#define KOKKOS_COPY_CUDA_LOCK_ARRAYS_TO_DEVICE() \
do { \
  CUDA_SAFE_CALL(cudaMemcpyToSymbol( \
        Kokkos::Impl::device_cuda_lock_arrays , \
        & Kokkos::Impl::host_cuda_lock_arrays , \
        sizeof(Kokkos::Impl::CudaLockArrays) ) ); \
} while (0)

#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
#define KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE()
#else
#define KOKKOS_ENSURE_CUDA_LOCK_ARRAYS_ON_DEVICE() KOKKOS_COPY_CUDA_LOCK_ARRAYS_TO_DEVICE()
#endif

#endif /* defined( __CUDACC__ ) */

#endif /* defined( KOKKOS_ENABLE_CUDA ) */

#endif /* #ifndef KOKKOS_CUDA_LOCKS_HPP */
