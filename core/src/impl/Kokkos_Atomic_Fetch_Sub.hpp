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

#if defined( KOKKOS_ATOMIC_HPP ) && ! defined( KOKKOS_ATOMIC_FETCH_SUB_HPP )
#define KOKKOS_ATOMIC_FETCH_SUB_HPP

namespace Kokkos {

//----------------------------------------------------------------------------

#if defined( KOKKOS_ATOMICS_USE_CUDA )

// Support for int, unsigned int, unsigned long long int, and float

__inline__ __device__
int atomic_fetch_sub( volatile int * const dest , const int val )
{ return atomicSub((int*)dest,val); }

__inline__ __device__
unsigned int atomic_fetch_sub( volatile unsigned int * const dest , const unsigned int val )
{ return atomicSub((unsigned int*)dest,val); }

template < typename T >
__inline__ __device__
T atomic_fetch_sub( volatile T * const dest ,
  typename Kokkos::Impl::enable_if< sizeof(T) == sizeof(int) , const T >::type val )
{
  union { int i ; T t ; } oldval , assume , newval ;

  oldval.t = *dest ;

  do {
    assume.i = oldval.i ;
    newval.t = assume.t - val ;
    oldval.i = atomicCAS( (int*)dest , assume.i , newval.i );
  } while ( assumed.i != oldval.i );

  return oldval.t ;
}

template < typename T >
__inline__ __device__
T atomic_fetch_sub( volatile T * const dest ,
  typename Kokkos::Impl::enable_if< sizeof(T) != sizeof(int) &&
                                    sizeof(T) == sizeof(unsigned long long int) , const T >::type val )
{
  union { unsigned long long int i ; T t ; } oldval , assume , newval ;

  oldval.t = *dest ;

  do {
    assume.i = oldval.i ;
    newval.t = assume.t - val ;
    oldval.i = atomicCAS( (unsigned long long int*)dest , assume.i , newval.i );
  } while ( assume.i != oldval.i );

  return oldval.t ;
}

//----------------------------------------------------------------------------

#elif defined(KOKKOS_ATOMICS_USE_GCC) || defined(KOKKOS_ATOMICS_USE_INTEL)

KOKKOS_INLINE_FUNCTION
int atomic_fetch_sub( volatile int * const dest , const int val )
{ return __sync_fetch_and_sub(dest,val); }

KOKKOS_INLINE_FUNCTION
long int atomic_fetch_sub( volatile long int * const dest , const long int val )
{ return __sync_fetch_and_sub(dest,val); }

#if defined( KOKKOS_ATOMICS_USE_GCC )

KOKKOS_INLINE_FUNCTION
unsigned int atomic_fetch_sub( volatile unsigned int * const dest , const unsigned int val )
{ return __sync_fetch_and_sub(dest,val); }

KOKKOS_INLINE_FUNCTION
unsigned long int atomic_fetch_sub( volatile unsigned long int * const dest , const unsigned long int val )
{ return __sync_fetch_and_sub(dest,val); }

#endif

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_sub( volatile T * const dest ,
  typename Kokkos::Impl::enable_if< sizeof(T) == sizeof(int) , const T >::type val )
{
  union { int i ; T t ; } assume , oldval , newval ;

  oldval.t = *dest ;

  do {
    assume.i = oldval.i ;
    newval.t = assume.t - val ;
    oldval.i = __sync_val_compare_and_swap( (int*) dest , assume.i , newval.i );
  } while ( assume.i != oldval.i );

  return oldval.t ;
}

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_sub( volatile T * const dest ,
  typename Kokkos::Impl::enable_if< sizeof(T) != sizeof(int) &&
                                    sizeof(T) == sizeof(long) , const T >::type val )
{
  union { long i ; T t ; } assume , oldval , newval ;

  oldval.t = *dest ;

  do {
    assume.i = oldval.i ;
    newval.t = assume.t - val ;
    oldval.i = __sync_val_compare_and_swap( (long*) dest , assume.i , newval.i );
  } while ( assume.i != oldval.i );

  return oldval.t ;
}

//----------------------------------------------------------------------------

#elif defined( KOKKOS_ATOMICS_USE_OMP31 )

template< typename T >
T atomic_fetch_sub( volatile T * const dest , const T val )
{
  T retval;
#pragma omp atomic capture
  {
    retval = dest[0];
    dest[0] -= val;
  }
  return retval;
}

#endif

//----------------------------------------------------------------------------

template < typename T >
KOKKOS_INLINE_FUNCTION
T atomic_fetch_SUB( volatile T * const dest ,
    typename ::Kokkos::Impl::enable_if<
                  ( sizeof(T) != 4 )
               && ( sizeof(T) != 8 )
            #if defined(KOKKOS_ENABLE_ASM) && !defined(__CUDA_ARCH__)
               && ( sizeof(T) != 16 )
            #endif
             , const T >::type& val )
{
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
  while( !HostSpace::lock_address( (void*) dest ) );
  T return_val = *dest;
  *dest = return_val - val;
  HostSpace::unlock_address( (void*) dest );
  return return_val;
#else
#endif
}

// Simpler version of atomic_fetch_sub without the fetch
template <typename T>
KOKKOS_INLINE_FUNCTION
void atomic_sub(volatile T * const dest, const T src) {
  atomic_fetch_sub(dest,src);
}

}

#include<impl/Kokkos_Atomic_Assembly_X86.hpp>
#endif


