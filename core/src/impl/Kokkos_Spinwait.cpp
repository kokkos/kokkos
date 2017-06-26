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
#include <impl/Kokkos_BitOps.hpp>

#include <Kokkos_Atomic.hpp>

#include <chrono>

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
/* KOKKOS_IMPL_NOP                                                          */
/*--------------------------------------------------------------------------*/
#if defined( KOKKOS_ENABLE_ASM )
  #if !defined( _WIN32 )
    #define KOKKOS_IMPL_NOP asm volatile("nop\n")
  #else
    #define KOKKOS_IMPL_NOP __asm__ __volatile__("nop\n")
  #endif
  #define KOKKOS_IMPL_NOP2  KOKKOS_IMPL_NOP  ; KOKKOS_IMPL_NOP
  #define KOKKOS_IMPL_NOP4  KOKKOS_IMPL_NOP2 ; KOKKOS_IMPL_NOP2
  #define KOKKOS_IMPL_NOP8  KOKKOS_IMPL_NOP4 ; KOKKOS_IMPL_NOP4
  #define KOKKOS_IMPL_NOP16 KOKKOS_IMPL_NOP8 ; KOKKOS_IMPL_NOP8
  #define KOKKOS_IMPL_NOP32 KOKKOS_IMPL_NOP16; KOKKOS_IMPL_NOP16
  #define KOKKOS_IMPL_NOP64 KOKKOS_IMPL_NOP32; KOKKOS_IMPL_NOP32
#else
  #define KOKKOS_IMPL_NOP   KOKKOS_IMPL_YIELD
  #define KOKKOS_IMPL_NOP2  KOKKOS_IMPL_YIELD
  #define KOKKOS_IMPL_NOP4  KOKKOS_IMPL_YIELD
  #define KOKKOS_IMPL_NOP8  KOKKOS_IMPL_YIELD
  #define KOKKOS_IMPL_NOP16 KOKKOS_IMPL_YIELD
  #define KOKKOS_IMPL_NOP32 KOKKOS_IMPL_YIELD
  #define KOKKOS_IMPL_NOP64 KOKKOS_IMPL_YIELD
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
  #define KOKKOS_IMPL_SLEEP( ns ) KOKKOS_IMPL_YIELD; KOKKOS_IMPL_NOP64
#endif

namespace {

inline void kokkos_impl_yield( int64_t duration )
{
  // clock might not be monotonic
  duration = duration > 0 ? duration : 0;

  // find most significant bit
  // log2 of duration
  const int i = Kokkos::Impl::bit_scan_reverse(duration);

  switch( i ) {
  case 0:
    break;
  case 1:
    break;
  case 2:
    break;
  case 3:
    KOKKOS_IMPL_NOP;
    break;
  case 4:
    KOKKOS_IMPL_NOP2;
    break;
  case 5:
    KOKKOS_IMPL_NOP4;
    break;
  case 6:
    KOKKOS_IMPL_NOP8;
    break;
  case 7:
    KOKKOS_IMPL_NOP16;
    break;
  case 8:
    KOKKOS_IMPL_YIELD;
    KOKKOS_IMPL_NOP32;
    break;
  case 9:
    KOKKOS_IMPL_YIELD;
    KOKKOS_IMPL_NOP64;
    break;
  default:
    // sleep for approximatly 1/2 as long as the current duration
    //KOKKOS_IMPL_SLEEP( (duration >> 1) );
    break;
  }
  KOKKOS_IMPL_PAUSE;
}


} // namespace

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void spinwait_while_equal( volatile int32_t & flag , const int32_t value )
{
  using ns        = std::chrono::nanoseconds;
  using clock     = std::chrono::high_resolution_clock;
  using timepoint = clock::time_point;

  Kokkos::store_fence();
  const timepoint start = clock::now();
  while ( value == flag ) {
    kokkos_impl_yield( std::chrono::duration_cast<ns>( clock::now() - start).count() );
  }
  Kokkos::load_fence();
}

void spinwait_until_equal( volatile int32_t & flag , const int32_t value )
{
  using ns        = std::chrono::nanoseconds;
  using clock     = std::chrono::high_resolution_clock;
  using timepoint = clock::time_point;

  Kokkos::store_fence();
  const timepoint start = clock::now();
  while ( value != flag ) {
    kokkos_impl_yield( std::chrono::duration_cast<ns>( clock::now() - start).count() );
  }
  Kokkos::load_fence();
}

void spinwait_while_equal( volatile int64_t & flag , const int64_t value )
{
  using ns        = std::chrono::nanoseconds;
  using clock     = std::chrono::high_resolution_clock;
  using timepoint = clock::time_point;

  Kokkos::store_fence();
  const timepoint start = clock::now();
  while ( value == flag ) {
    kokkos_impl_yield( std::chrono::duration_cast<ns>( clock::now() - start).count() );
  }
  Kokkos::load_fence();
}

void spinwait_until_equal( volatile int64_t & flag , const int64_t value )
{
  using ns        = std::chrono::nanoseconds;
  using clock     = std::chrono::high_resolution_clock;
  using timepoint = clock::time_point;

  Kokkos::store_fence();
  const timepoint start = clock::now();
  while ( value != flag ) {
    kokkos_impl_yield( std::chrono::duration_cast<ns>( clock::now() - start).count() );
  }
  Kokkos::load_fence();
}

} /* namespace Impl */
} /* namespace Kokkos */

#else
void KOKKOS_CORE_SRC_IMPL_SPINWAIT_PREVENT_LINK_ERROR() {}
#endif

