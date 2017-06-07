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

namespace Kokkos {
namespace Impl {

struct CudaLockArrays {
  int* atomic;
  int* threadid;
  int n;
};

extern Kokkos::Impl::CudaLockArrays host_cuda_lock_arrays ;

void initialize_host_cuda_lock_arrays();
void finalize_host_cuda_lock_arrays();

} // namespace Impl
} // namespace Kokkos

#if defined( __CUDACC__ )

__device__ __constant__
#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
extern
#endif
Kokkos::Impl::CudaLockArrays kokkos_impl_cuda_lock_arrays ;

#define CUDA_SPACE_ATOMIC_MASK 0x1FFFF
#define CUDA_SPACE_ATOMIC_XOR_MASK 0x15A39

namespace Kokkos {
namespace Impl {

__device__ inline
bool lock_address_cuda_space(void* ptr) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & CUDA_SPACE_ATOMIC_MASK;
  return (0 == atomicCAS(&kokkos_impl_cuda_lock_arrays.atomic[offset],0,1));
}

__device__ inline
void unlock_address_cuda_space(void* ptr) {
  size_t offset = size_t(ptr);
  offset = offset >> 2;
  offset = offset & CUDA_SPACE_ATOMIC_MASK;
  atomicExch( &kokkos_impl_cuda_lock_arrays.atomic[ offset ], 0);
}

inline
void copy_cuda_lock_arrays_to_device() {
  cudaMemcpyToSymbol( kokkos_impl_cuda_lock_arrays ,
                      & Kokkos::Impl::host_cuda_lock_arrays ,
                      sizeof(CudaLockArrays) );
}

inline
void ensure_cuda_lock_arrays_on_device() {
#ifndef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
  copy_cuda_lock_arrays_to_device();
#endif
}

} // namespace Impl
} // namespace Kokkos

#endif /* defined( __CUDACC__ ) */

#endif /* defined( KOKKOS_ENABLE_CUDA ) */

#endif /* #ifndef KOKKOS_CUDA_LOCKS_HPP */
