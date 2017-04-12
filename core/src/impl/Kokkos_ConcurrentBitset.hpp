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

#ifndef KOKKOS_CONCURRENTBITSET_HPP
#define KOKKOS_CONCURRENTBITSET_HPP

#include <stdint.h>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_ClockTic.hpp>

namespace Kokkos {
namespace Impl {

struct concurrent_bitset {
public:

  // 32 bits per integer value

  enum : uint32_t { bits_per_int_lg2  = 5 };
  enum : uint32_t { bits_per_int_mask = ( 1 << bits_per_int_lg2 ) - 1 };

  // Array length alignment to 8 integers, 32 bytes

  enum : uint32_t { align_size  = 8 /* 32 bytes, and power of two */ };
  enum : uint32_t { align_mask  = align_size - 1 };

  // uint state { ( bit_count_lg2 << state_shift ) | used_count }

  enum : uint32_t { state_shift = 26 };
  enum : uint32_t { state_mask  = ( 1 << state_shift ) - 1 };

  // Buffer is uint32_t[ buffer_length( bit_count_lg2 ) ]
  //   [ uint32_t state , uint32_t bits[*] ]

  //  Maximum bit count is 33 million to enable:
  //  - bit_count_lg2 can be stored in the upper 5 bits
  //  - accept up to 33 million concurrent calls to 'acquire'
  //    before risking an overflow race condition on a full bitset.

  enum : uint32_t { max_bit_count_lg2 = 25 };
  enum : uint32_t { max_bit_count     = 1u << max_bit_count_lg2 };

  /**\brief  Buffer is int[ bitset_storage_length( bit_count_lg2 ) ]
   *
   *  Requires: bit_count_lg2 <= max_bit_count_lg2
   */
  KOKKOS_INLINE_FUNCTION static constexpr
  uint32_t buffer_length( uint32_t bit_count_lg2 ) noexcept
    {
      return bit_count_lg2 <= max_bit_count_lg2
           ? ( ( 1 + ( 1 << ( bit_count_lg2 > bits_per_int_lg2
                            ? bit_count_lg2 - bits_per_int_lg2 : 0 ) )
                 + align_mask
               ) & ~align_mask )
           : 0 ;
    }

  /**\brief  Initialize bitset buffer */
  KOKKOS_INLINE_FUNCTION static
  void buffer_init( uint32_t bit_count_lg2 , uint32_t * const buffer ) noexcept
    {
      const uint32_t len = buffer_length( bit_count_lg2 );

      for ( uint32_t i = 0 ; i < len ; ++i ) buffer[i] = 0 ;

      buffer[0] = bit_count_lg2 << state_shift ;

      Kokkos::memory_fence();
    }

  KOKKOS_FORCEINLINE_FUNCTION static
  uint32_t capacity_lg2( uint32_t const * const buffer ) noexcept
    { return buffer[0] >> state_shift ; }

  KOKKOS_FORCEINLINE_FUNCTION static
  uint32_t capacity( uint32_t const * const buffer ) noexcept
    { return 1u << ( buffer[0] >> state_shift ); }

  /**\brief  Claim any bit within the bitset.
   *
   *  Return : ( which_bit , bit_count )
   *
   *  if success then
   *    bit_count is the atomic-count of claimed > 0
   *    which_bit is the claimed bit >= 0
   *  else if attempt failed due to filled buffer
   *    bit_count == which_bit == -1
   *  else if attempt failed due to bit_count_lg2 does not match buffer
   *    bit_count == which_bit == -2
   *  else if attempt failed due to max_bit_count_lg2 < bit_count_lg2
   *    bit_count == which_bit == -3
   *  endif
   *
   *  Recommended to have bit_hint = Kokkos::Impl::clock_tic()
   */
  KOKKOS_INLINE_FUNCTION static
  Kokkos::pair<int,int>
  acquire( uint32_t volatile * const buffer
         , uint32_t bit_hint      = 0 /* performance hint */
         , uint32_t bit_count_lg2 = 0 /* if > 0 verify */
         ) noexcept
    {
      typedef Kokkos::pair<int,int> type ;

      if ( max_bit_count_lg2 < bit_count_lg2 ) return type(-3,-3);

      // Use potentially two fetch_add to avoid CAS loop.
      // Could generate "racing" failure-to-acquire
      // when is full at the atomic_fetch_add(+1)
      // then a release occurs before the atomic_fetch_add(-1).

      const uint32_t state = (uint32_t)
        Kokkos::atomic_fetch_add( (volatile int *) buffer , 1 );

      const uint32_t state_bit_count_lg2 = state >> state_shift ;
      const uint32_t state_bit_count     = 1 << state_bit_count_lg2 ;
      const uint32_t state_bit_used      = state & state_mask ;

      const uint32_t full_error = state_bit_count <= state_bit_used ;

      const uint32_t state_error =
        bit_count_lg2 && ( bit_count_lg2 != state_bit_count_lg2 );

      if ( full_error || state_error ) {
        Kokkos::atomic_fetch_add( (volatile int *) buffer , -1 );
        return full_error ? type(-1,-1) : type(-2,-2);
      }

      // Do not update bit until count is visible:

      Kokkos::memory_fence();

      const uint32_t word_count = state_bit_count >> bits_per_int_lg2 ;

      // There is a zero bit available somewhere,
      // now find the (first) available bit and set it.

      bit_hint &= ( state_bit_count - 1 );

      uint32_t i = bit_hint >> bits_per_int_lg2 ;

      bit_hint &= bits_per_int_mask ;

      int j = bit_hint ;

      while(1) {

        const uint32_t bit = 1 << j ;
        const uint32_t prev = Kokkos::atomic_fetch_or( buffer + 1 + i , bit );

        if ( ! ( prev & bit ) ) {
          // Successfully claimed 'result.first' by
          // atomically setting that bit.
          return type( ( i << bits_per_int_lg2 ) | uint32_t(j)
                     , ( state & state_mask ) + 1 );
        }

        // Failed race to set the selected bit
        // Find a new bit to try.

        j = Kokkos::Impl::bit_first_zero( prev );

        if ( j < 0 ) {

          if ( word_count == ++i ) i = 0 ;

          j = bit_hint ;
        }
      }
    }

  /**\brief
   *
   *  Requires: 'bit' previously acquired and has not yet been released.
   *
   *  Returns:
   *    0 <= used count after successful release
   *    -1 bit was already released
   *    -2 bit was not in range
   *    -3 bit_count_lg2 error
   */
  KOKKOS_INLINE_FUNCTION static
  int
  release( uint32_t volatile * const buffer
         , uint32_t bit               /* which bit */
         , uint32_t bit_count_lg2 = 0 /* if > 0 verify */
         ) noexcept
    {
      const uint32_t state_bit_count_lg2 = (*buffer) >> state_shift ;

      if ( bit_count_lg2 && ( bit_count_lg2 != state_bit_count_lg2 ) ) {
        return -3 ;
      }

      const uint32_t state_bit_count = 1 << state_shift ;

      if ( state_bit_count <= bit ) { return -2 ; }

      const uint32_t mask = 1u << ( bit & bits_per_int_mask );
      const uint32_t prev =
        Kokkos::atomic_fetch_and( buffer + ( bit >> bits_per_int_lg2 ) + 1
                                , ~mask
                                );

      if ( ! ( prev & mask ) ) { return -1 ; }

      // Do not update count until bit clear is visible
      Kokkos::memory_fence();

      const int count =
        Kokkos::atomic_fetch_add( (volatile int *) buffer , -1 );

      return ( count & state_mask ) - 1 ;
    }
};

}} // namespace Kokkos::Impl

#endif /* #ifndef KOKKOS_CONCURRENTBITSET_HPP */

