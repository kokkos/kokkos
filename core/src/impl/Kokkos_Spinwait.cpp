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

#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_Spinwait.hpp>
#include <impl/Kokkos_BitOps.hpp>

#if defined( KOKKOS_ENABLE_STDTHREAD )
  #include <thread>
#elif !defined( _WIN32 )
  #include <sched.h>
  #include <time.h>
#else
  #include <process.h>
  #include <winsock2.h>
  #include <windows.h>
#endif

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {
namespace {

void host_thread_yield( const uint32_t i )
{
  const int c = Kokkos::Impl::bit_scan_reverse( i );

  switch( c ) {
  default:

    // Attempt to put the thread to sleep for 'c' milliseconds

    {
      #if defined( KOKKOS_ENABLE_STDTHREAD )
        std::this_thread::sleep_for( std::chrono::nanoseconds( c * 1000 ) )
      #elif !defined( _WIN32 )
        timespec req ;
        req.tv_sec  = 0 ;
        req.tv_nsec = 1000 * c ;
        nanosleep( &req, nullptr );
      #else /* defined( _WIN32 ) IS Microsoft Windows */
        Sleep(c);
      #endif
    }
    // [[fallthrough]]; // suppress fallthrough warning

  case 14: /* 16k attempts before yielding */

#if ! defined( KOKKOS_ENABLE_ASM )

  case 13: // [[fallthrough]]; // suppress fallthrough warning
  case 12: // [[fallthrough]]; // suppress fallthrough warning
  case 11: // [[fallthrough]]; // suppress fallthrough warning
  case 10: // [[fallthrough]]; // suppress fallthrough warning
  case 9 : // [[fallthrough]]; // suppress fallthrough warning
  case 8 : // [[fallthrough]]; // suppress fallthrough warning
  case 7 : // [[fallthrough]]; // suppress fallthrough warning
  case 6 : // [[fallthrough]]; // suppress fallthrough warning
  case 5 : // [[fallthrough]]; // suppress fallthrough warning
  case 4 : // [[fallthrough]]; // suppress fallthrough warning
  case 3 : // [[fallthrough]]; // suppress fallthrough warning
  case 2 : // [[fallthrough]]; // suppress fallthrough warning
  case 1 : // [[fallthrough]]; // suppress fallthrough warning
  case 0 : // [[fallthrough]]; // suppress fallthrough warning

#endif /* #if defined( KOKKOS_ENABLE_ASM ) */

    // Attempt to yield thread resources to runtime

    #if defined( KOKKOS_ENABLE_STDTHREAD )
      std::this_thread::yield();
    #elif !defined( _WIN32 )
      sched_yield();
    #else /* defined( _WIN32 ) IS Microsoft Windows */
      YieldProcessor();
    #endif
    // [[fallthrough]]; // suppress fallthrough warning

#if defined( KOKKOS_ENABLE_ASM )

  case 13: // [[fallthrough]]; // suppress fallthrough warning
  case 12: // [[fallthrough]]; // suppress fallthrough warning
  case 11: // [[fallthrough]]; // suppress fallthrough warning
  case 10: // [[fallthrough]]; // suppress fallthrough warning
  case 9 : // [[fallthrough]]; // suppress fallthrough warning
  case 8 : // [[fallthrough]]; // suppress fallthrough warning
  case 7 : // [[fallthrough]]; // suppress fallthrough warning
  case 6 : // [[fallthrough]]; // suppress fallthrough warning
  case 5 : // [[fallthrough]]; // suppress fallthrough warning
  case 4 : // [[fallthrough]]; // suppress fallthrough warning

    // Insert a few no-ops to quiet the thread:

    for ( int k = 0 ; k < c ; ++k ) {
      #if !defined( _WIN32 ) /* IS NOT Microsoft Windows */
        asm volatile("nop\n");
      #else /* IS Microsoft Windows */
        __asm__ __volatile__("nop\n");
      #endif
    }
    // [[fallthrough]]; // suppress fallthrough warning
  case 3 : // [[fallthrough]]; // suppress fallthrough warning
  case 2 : // [[fallthrough]]; // suppress fallthrough warning
  case 1 : // [[fallthrough]]; // suppress fallthrough warning
  case 0 : // [[fallthrough]]; // suppress fallthrough warning

    // Insert memory pause

    #if !defined( _WIN32 ) /* IS NOT Microsoft Windows */
      #if !defined( __arm__ ) && !defined( __aarch64__ )
        asm volatile("pause\n":::"memory");
      #endif
    #else /* IS Microsoft Windows */
      __asm__ __volatile__("pause\n":::"memory");
    #endif

#endif /* #if defined( KOKKOS_ENABLE_ASM ) */

  }
}

}}} // namespace Kokkos::Impl::{anonymous}

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void spinwait_while_equal( volatile int32_t & flag , const int32_t value )
{
  Kokkos::store_fence();
  uint32_t i = 0 ; while( value == flag ) host_thread_yield(++i);
  Kokkos::load_fence();
}

void spinwait_until_equal( volatile int32_t & flag , const int32_t value )
{
  Kokkos::store_fence();
  uint32_t i = 0 ; while( value != flag ) host_thread_yield(++i);
  Kokkos::load_fence();
}

void spinwait_while_equal( volatile int64_t & flag , const int64_t value )
{
  Kokkos::store_fence();
  uint32_t i = 0 ; while( value == flag ) host_thread_yield(++i);
  Kokkos::load_fence();
}

void spinwait_until_equal( volatile int64_t & flag , const int64_t value )
{
  Kokkos::store_fence();
  uint32_t i = 0 ; while( value != flag ) host_thread_yield(++i);
  Kokkos::load_fence();
}

} /* namespace Impl */
} /* namespace Kokkos */

#else
void KOKKOS_CORE_SRC_IMPL_SPINWAIT_PREVENT_LINK_ERROR() {}
#endif

