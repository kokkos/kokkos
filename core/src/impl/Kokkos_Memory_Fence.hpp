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

#if defined( KOKKOS_ATOMIC_HPP ) && ! defined( KOKKOS_MEMORY_FENCE )
#define KOKKOS_MEMORY_FENCE
namespace Kokkos {

//----------------------------------------------------------------------------

KOKKOS_FORCEINLINE_FUNCTION
void memory_fence()
{
#if defined( KOKKOS_ATOMICS_USE_CUDA )
  __threadfence();
#elif defined( KOKKOS_ATOMICS_USE_GCC ) || \
      ( defined( KOKKOS_COMPILER_NVCC ) && defined( KOKKOS_ATOMICS_USE_INTEL ) )
  __sync_synchronize();
#elif defined( KOKKOS_ATOMICS_USE_INTEL )
  _mm_mfence();
#elif defined( KOKKOS_ATOMICS_USE_OMP31 )
  #pragma omp flush
#elif defined( KOKKOS_ATOMICS_USE_WINDOWS )
  MemoryBarrier();
#else
 #error "Error: memory_fence() not defined"
#endif
}

//////////////////////////////////////////////////////
// store_fence()
//
// If possible use a store fence on the architecture, if not run a full memory fence

KOKKOS_FORCEINLINE_FUNCTION
void store_fence()
{
#if defined( KOKKOS_ENABLE_ASM ) && defined( KOKKOS_USE_ISA_X86_64 )
  asm volatile (
	"sfence" ::: "memory"
  	);
#else
  memory_fence();
#endif
}

//////////////////////////////////////////////////////
// load_fence()
//
// If possible use a load fence on the architecture, if not run a full memory fence

KOKKOS_FORCEINLINE_FUNCTION
void load_fence()
{
#if defined( KOKKOS_ENABLE_ASM ) && defined( KOKKOS_USE_ISA_X86_64 )
  asm volatile (
	"lfence" ::: "memory"
  	);
#else
  memory_fence();
#endif
}

} // namespace kokkos

#endif


