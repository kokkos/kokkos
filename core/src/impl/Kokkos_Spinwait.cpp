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

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )

#include <impl/Kokkos_Spinwait.hpp>

#include <Kokkos_Atomic.hpp>

#include <algorithm>

#if defined( KOKKOS_ENABLE_STDTHREAD )
  #include <thread>
#elif !defined( _WIN32 )
  #include <sched.h>
  #include <time.h>
#endif

/*--------------------------------------------------------------------------*/
/* KOKKOS_IMPL_YIELD                                                        */
/*--------------------------------------------------------------------------*/

#if defined( KOKKOS_ENABLE_STDTHREAD )
  #define KOKKOS_IMPL_YIELD std::this_thread::yield()
#elif !defined( _WIN32 )
  #define KOKKOS_IMPL_YIELD sched_yield()
#else
  #define KOKKOS_IMPL_YIELD
#endif

/*--------------------------------------------------------------------------*/
/* KOKKOS_IMPL_PAUSE                                                        */
/*--------------------------------------------------------------------------*/
#if defined( KOKKOS_ENABLE_ASM ) && !defined( _WIN32 ) && !defined( __arm__ ) && !defined( __aarch64__ )
  #define KOKKOS_IMPL_PAUSE asm volatile("pause\n":::"memory")
#else
  #define KOKKOS_IMPL_PAUSE
#endif

/*--------------------------------------------------------------------------*/
/* KOKKOS_IMPL_SLEEP                                                        */
/*--------------------------------------------------------------------------*/
#if defined( KOKKOS_ENABLE_STDTHREAD )
  #define KOKKOS_IMPL_SLEEP( ns ) std::this_thread::sleep_for( std::chrono::nanoseconds( ns ) )
#elif !defined( _WIN32 )
  #define KOKKOS_IMPL_SLEEP( ns )                                \
  {                                                              \
    timespec req;                                                \
    req.tv_sec  = ns < 1000000000 ? 0        : ns / 1000000000 ; \
    req.tv_nsec = ns < 1000000000 ? (long)ns : ns % 1000000000 ; \
    nanosleep( &req, nullptr );                                  \
  }
#else
  #define KOKKOS_IMPL_SLEEP( ns ) KOKKOS_IMPL_YIELD;
#endif

namespace {

inline void kokkos_impl_yield( const uint32_t i )
{
  if ( i < 1024u ) {
    KOKKOS_IMPL_YIELD;
  }
  else {
    KOKKOS_IMPL_SLEEP( 1000 );
  }
  KOKKOS_IMPL_PAUSE;
}


} // namespace

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void spinwait_while_equal( volatile int32_t & flag , const int32_t value )
{
  Kokkos::store_fence();
  uint32_t i=0;
  while ( value == flag ) {
    kokkos_impl_yield( ++i );
  }
  Kokkos::load_fence();
}

void spinwait_until_equal( volatile int32_t & flag , const int32_t value )
{
  Kokkos::store_fence();
  uint32_t i=0;
  while ( value != flag ) {
    kokkos_impl_yield( ++i );
  }
  Kokkos::load_fence();
}

void spinwait_while_equal( volatile int64_t & flag , const int64_t value )
{
  Kokkos::store_fence();
  uint32_t i=0;
  while ( value == flag ) {
    kokkos_impl_yield( ++i );
  }
  Kokkos::load_fence();
}

void spinwait_until_equal( volatile int64_t & flag , const int64_t value )
{
  Kokkos::store_fence();
  uint32_t i=0;
  while ( value != flag ) {
    kokkos_impl_yield( ++i );
  }
  Kokkos::load_fence();
}

} /* namespace Impl */
} /* namespace Kokkos */

#else
void KOKKOS_CORE_SRC_IMPL_SPINWAIT_PREVENT_LINK_ERROR() {}
#endif

