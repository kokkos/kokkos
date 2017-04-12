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

#ifndef KOKKOS_MEMORYPOOLV2_HPP
#define KOKKOS_MEMORYPOOLV2_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_ConcurrentBitset.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {
namespace Experimental {

template< typename DeviceType >
class MemoryPoolv2 {
private:

  typedef typename Kokkos::Impl::concurrent_bitset CB ;

  enum : uint32_t { bits_per_int_lg2  = CB::bits_per_int_lg2 };
  enum : uint32_t { int_align_mask    = CB::align_mask };

  enum : uint32_t { state_shift       = CB::state_shift };
  enum : uint32_t { state_mask        = CB::state_mask };
  enum : uint32_t { max_bit_count_lg2 = CB::max_bit_count_lg2 };
  enum : uint32_t { max_bit_count     = CB::max_bit_count };

  /* Minimum block size for an allocation */
  enum : uint32_t { LG2_MIN_BLOCK_SIZE = 6  /* 64bytes == ( 1 << 6 ) */ };

  /*  Each superblock has a concurrent bitset state
   *  which is an array of unsigned integers.
   *    [ { block_count_lg2  : state_shift bits
   *      , used_block_count : ( 32 - state_shift ) bits
   *      }
   *    , { block allocation bit set }* ]
   *
   *  As superblocks are assigned (allocated) to a block size
   *  and released (deallocated) back to empty the superblock state
   *  is concurrently updated.
   */

  typedef typename DeviceType::memory_space base_memory_space ;

  enum { accessible =
           Kokkos::Impl::MemorySpaceAccess< Kokkos::HostSpace 
                                          , base_memory_space >::accessible };

  typedef Kokkos::Impl::SharedAllocationTracker Tracker ;
  typedef Kokkos::Impl::SharedAllocationRecord
    < base_memory_space >  Record ;

  Tracker    m_tracker ;
  uint32_t * m_sb_state_array ;
  uint32_t   m_sb_state_size ;
  uint32_t   m_sb_size_lg2 ;
  uint32_t   m_sb_count ;
  uint32_t   m_sb_hint_offset ;
  uint32_t   m_sb_data_offset ;

public:

  //--------------------------------------------------------------------------

  struct ZeroFunctor {

    uint32_t * array ;

    KOKKOS_INLINE_FUNCTION
    void operator()( int i ) const noexcept { array[i] = 0 ; }

    ZeroFunctor( const ZeroFunctor & ) = default ;

    ZeroFunctor( typename DeviceType::execution_space const & space
               , unsigned * a , unsigned n )
      : array(a)
      {
        typedef Kokkos::RangePolicy< typename DeviceType::execution_space >
            Policy ;

        const Kokkos::Impl::ParallelFor< ZeroFunctor , Policy >
          closure( *this , Policy( 0 , n ) );
        closure.execute();
        space.fence();
      }
  };

  //--------------------------------------------------------------------------

  void print_state( std::ostream & s ) const
    {
      Kokkos::HostSpace host ;

      const size_t alloc_size = m_sb_hint_offset * sizeof(uint32_t);

      uint32_t * const sb_state_array = 
        accessible ? m_sb_state_array : (uint32_t *) host.allocate(alloc_size);

      if ( ! accessible ) {
        Kokkos::Impl::DeepCopy< Kokkos::HostSpace , base_memory_space >
          ( sb_state_array , m_sb_state_array , alloc_size );
      }

      const uint32_t * sb_state_ptr = sb_state_array ;

      s << "pool_size(" << ( ( 1 << m_sb_size_lg2 ) * m_sb_count ) << ")"
        << " superblock_size(" << ( 1 << m_sb_size_lg2 ) << ")" << std::endl ;

      for ( unsigned i = 0 ; i < m_sb_count
          ; ++i , sb_state_ptr += m_sb_state_size ) {

        const unsigned block_count_lg2 = (*sb_state_ptr) >> state_shift ;

        if ( block_count_lg2 ) {

          const unsigned block_size_lg2 = m_sb_size_lg2 - block_count_lg2 ;
          const unsigned block_count    = 1 << block_count_lg2 ;
          const unsigned block_used     = (*sb_state_ptr) & state_mask ;

          s << "Superblock[ " << i << " / " << m_sb_count << " ] {"
            << " block_size(" << ( 1 << block_size_lg2 ) << ")"
            << " block_count( " << block_used
            << " / " << block_count  << " )"
            << std::endl ;
        }
      }

      if ( ! accessible ) {
        host.deallocate( sb_state_array, alloc_size );
      }
    }

  KOKKOS_INLINE_FUNCTION
  size_t capacity() const noexcept
    { return ( 1LU << m_sb_size_lg2 ) * m_sb_count ; }

  struct usage_statistics {
    size_t capacity_bytes ;       ///<  Capacity in bytes
    size_t superblock_bytes ;     ///<  Superblocks capacity in bytes
    size_t capacity_superblocks ; ///<  Number of superblocks
    size_t consumed_superblocks ; ///<  Superblocks assigned to allocations
    size_t consumed_blocks ;  ///<  Number of allocations
    size_t consumed_bytes ;   ///<  Bytes allocated
    size_t reserved_blocks ;  ///<  Unallocated blocks in assigned superblocks
    size_t reserved_bytes ;   ///<  Unallocated bytes in assigned superblocks
  };

  void get_usage_statistics( usage_statistics & stats ) const
    {
      Kokkos::HostSpace host ;

      const size_t alloc_size = m_sb_hint_offset * sizeof(uint32_t);

      uint32_t * const sb_state_array = 
        accessible ? m_sb_state_array : (uint32_t *) host.allocate(alloc_size);

      if ( ! accessible ) {
        Kokkos::Impl::DeepCopy< Kokkos::HostSpace , base_memory_space >
          ( sb_state_array , m_sb_state_array , alloc_size );
      }

      stats.superblock_bytes = ( 1LU << m_sb_size_lg2 );
      stats.capacity_bytes = stats.superblock_bytes * m_sb_count ;
      stats.capacity_superblocks = m_sb_count ;
      stats.consumed_superblocks = 0 ;
      stats.consumed_blocks = 0 ;
      stats.consumed_bytes = 0 ;
      stats.reserved_blocks = 0 ;
      stats.reserved_bytes = 0 ;

      const uint32_t * sb_state_ptr = sb_state_array ;

      for ( unsigned i = 0 ; i < m_sb_count
          ; ++i , sb_state_ptr += m_sb_state_size ) {

        const unsigned block_count_lg2 = (*sb_state_ptr) >> state_shift ;

        if ( block_count_lg2 ) {
          const unsigned block_count    = 1u << block_count_lg2 ;
          const unsigned block_size_lg2 = m_sb_size_lg2 - block_count_lg2 ;
          const unsigned block_size     = 1u << block_size_lg2 ;
          const unsigned block_used     = (*sb_state_ptr) & state_mask ;

          stats.consumed_superblocks++ ;
          stats.consumed_blocks += block_used ;
          stats.consumed_bytes  += block_used * block_size ;
          stats.reserved_blocks += block_count - block_used ;
          stats.reserved_bytes  += (block_count - block_used ) * block_size ;
        }
      }

      if ( ! accessible ) {
        host.deallocate( sb_state_array, alloc_size );
      }
    }

  //--------------------------------------------------------------------------

  MemoryPoolv2() = default ;
  MemoryPoolv2( MemoryPoolv2 && ) = default ;
  MemoryPoolv2( const MemoryPoolv2 & ) = default ;
  MemoryPoolv2 & operator = ( MemoryPoolv2 && ) = default ;
  MemoryPoolv2 & operator = ( const MemoryPoolv2 & ) = default ;

  /**\brief  Allocate a memory pool from 'memspace'.
   *
   *  The memory pool will have at least 'min_alloc_size' bytes
   *  of memory to allocate divided among superblocks of at least
   *  'min_superblock_size' bytes.  A single allocation must fit
   *  within a single superblock, so 'min_superblock_size' must be
   *  at least as large as the maximum single allocation.
   *  Both 'min_alloc_size' and 'min_superblock_size' are rounded up to the
   *  smallest power-of-two value that contains the corresponding sizes.
   *  Individual allocations will always consume a block of memory that
   *  is also a power-of-two.  These roundings are made to enable
   *  significant runtime performance improvements.
   *
   *  TODO: Extend the "superblock hint" to an array with length as a
   *        fraction of the number of superblocks / number of block sizes.
   */
  MemoryPoolv2( const base_memory_space & memspace
              , const size_t   min_alloc_size
              , const unsigned min_superblock_size = 1 << 18 /* 256k */ )
    : m_tracker()
    , m_sb_state_array(0)
    , m_sb_state_size(0)
    , m_sb_size_lg2(0)
    , m_sb_count(0)
    , m_sb_hint_offset(0)
    , m_sb_data_offset(0)
    {
      // superblock size is power of two that can hold min_superblock_size

      m_sb_size_lg2 = Kokkos::Impl::bit_scan_reverse( min_superblock_size );

      if ( ( 1u << m_sb_size_lg2 ) < min_superblock_size ) ++m_sb_size_lg2 ;

      // At least two minimum size blocks in a superblock = 128 bytes
      if ( m_sb_size_lg2 < LG2_MIN_BLOCK_SIZE + 1 ) {
        m_sb_size_lg2 = LG2_MIN_BLOCK_SIZE + 1 ;
      }

      // Maximum superblock size 

      if ( max_bit_count_lg2 + LG2_MIN_BLOCK_SIZE < m_sb_size_lg2 ) {
        fprintf( stderr
               , "Kokkos MemoryPoolv2 maximum superblock size %ld\n"
               , ( 1L << ( max_bit_count_lg2 + LG2_MIN_BLOCK_SIZE ) )
               );
      }

      // Any superblock can be assigned to the smallest size block
      // Maximum number of blocks == 1 << ( m_sb_size_lg2 - LG2_MIN_BLOCK_SIZE )

      m_sb_state_size =
        CB::buffer_length( m_sb_size_lg2 - LG2_MIN_BLOCK_SIZE );

      // Hint array is one unsigned per block size

      const unsigned hint_array_size =
        ( ( 1 + m_sb_size_lg2 - LG2_MIN_BLOCK_SIZE ) + int_align_mask ) & ~int_align_mask ;

      // number of superblocks is multiple of superblock size that
      // can hold min_alloc_size.

      const uint32_t sb_size_mask = ( 1u << m_sb_size_lg2 ) - 1 ;

      m_sb_count = ( min_alloc_size + sb_size_mask ) >> m_sb_size_lg2 ;

      m_sb_hint_offset = m_sb_count * m_sb_state_size ;
      m_sb_data_offset = m_sb_hint_offset + hint_array_size ;

      // Allocation:

      const size_t alloc_size =
        ( m_sb_data_offset * sizeof(uint32_t) ) +
        ( m_sb_count << m_sb_size_lg2 );

      Record * rec = Record::allocate( memspace , "MemoryPoolv2" , alloc_size );

      m_tracker.assign_allocated_record_to_uninitialized( rec );

      m_sb_state_array = (unsigned *) rec->data();

      ZeroFunctor( typename DeviceType::execution_space()
                 , m_sb_state_array
                 , m_sb_data_offset );
    }

  //--------------------------------------------------------------------------

private:

  /* Given a size 'n' get the block size in which it can be allocated.
   * Restrict lower bound to minimum block size.
   */
  KOKKOS_FORCEINLINE_FUNCTION static
  int get_block_size_lg2( size_t n ) noexcept
    {
      int i = Kokkos::Impl::bit_scan_reverse( (unsigned) n );

      if ( (1u<<i) < n ) ++i ;

      return i < LG2_MIN_BLOCK_SIZE ? LG2_MIN_BLOCK_SIZE : i ;
    }

public:

  //--------------------------------------------------------------------------
  /**\brief  Allocate a block of memory that is at least 'alloc_size'
   *
   *  The block of memory is aligned to the minimum block size,
   *  currently is 64 bytes, will never be less than 32 bytes.
   *
   *  If concurrent allocations and deallocations are taking place
   *  then a single allocation attempt may fail due to lack of available space.
   *  The allocation attempt will retry up to 'retry_limit' times.
   */
  KOKKOS_FUNCTION
  void * allocate( size_t alloc_size
                 , uint32_t retry_limit = 0 ) const noexcept
    {
      enum : uint32_t { hint_lock = ~0u };

      void * p = 0 ;

      if ( alloc_size <= (1UL << m_sb_size_lg2) ) {

        // Allocation will fit within a superblock
        // that has block sizes ( 1 << block_size_lg2 )

        const uint32_t block_size_lg2  = get_block_size_lg2( alloc_size );
        const uint32_t block_count_lg2 = m_sb_size_lg2 - block_size_lg2 ;
        const uint32_t block_count     = 1u << block_count_lg2 ;

        //  Limit to trigger acquisition of an empty superblock.
        //  For every full word required to hold a block reduce by two.
        //  This corresponds to 94% full (30/32).
        //
        //  If block_count is less than a word then the allocation
        //  is a very large portion of the superblock and
        //  the concurrent bitset operation does not iterate words
        //  so is best to fill superblock to 100%.

        const uint32_t block_count_threshold =
          block_count - ( block_count >> ( bits_per_int_lg2 - 1 ) );

        // Hint for which superblock can support this allocation:
        // This superblock should be (not guaranteed due to concurrency)
        // of the given blocks size and have unclaimed blocks.

        volatile uint32_t * const hint_sb_id_ptr = m_sb_state_array +
          m_sb_hint_offset + ( block_size_lg2 - LG2_MIN_BLOCK_SIZE );

// printf("allocate(%ld) block_size(%d) block_count(%d)\n"
//       , alloc_size , ( 1 << block_size_lg2 ) , block_count );

        // Fast query clock register 'tic' to pseudo-randomize
        // the guess for which block within a superblock should
        // be claimed.  If not available then a search occurs.

        const uint32_t tic = (uint32_t)( Kokkos::Impl::clock_tic()
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA )
          // Spread out potentially concurrent access
          // by threads within a warp or thread block.
          + ( threadIdx.x + blockDim.x * threadIdx.y )
#endif
          );

        const uint32_t sb_id_tic = tic % m_sb_count ;

        while(1) {

          // When 0 <= sb_id then require:
          //   sb_state_array == m_sb_state_array + ( id * m_sb_state_size )

          int sb_id = -1 ; // superblock index
          volatile uint32_t * sb_state_array = 0 ; // superblock state

          //------------------------------------------------------------------
          // Get a valid superblock from which to obtain a block
          //------------------------------------------------------------------
          // Superblock id hint is locked when a thread is attempting to
          // claim an empty superblock for an allocation of this block size.

          const uint32_t hint_sb_id = *hint_sb_id_ptr ;

          if ( 0 != p ) {

            // This thread has allocated and filled its superblock to
            // the threshold. This thread is attempting to find a
            // superblock that is of the required block size and
            // is below the threshold to update the hint.
            // to that superblock.

            if ( hint_lock == hint_sb_id ) {
              // Another thread is updating the hint.
              // This thread's work is done.
              break ;
            }
          }

          //------------------------------------------------------------------
          //  Look for a partially full superblock of this block size.

          {
            //  If the allocation has already succeeded ( 0 != p )
            //  then look for a superblock below the threshold,
            //  otherwise look for a non-full superblock.

            const uint32_t target_count =
              ( 0 == p ) ? block_count : block_count_threshold ;

            //  If the hint is not locked it is supposed to identify
            //  a usable superblock so start searching there.
            //  If the hint is locked start searching at a "random" location.

            int id = ( hint_lock != hint_sb_id ) ? hint_sb_id : sb_id_tic ;

            sb_state_array = m_sb_state_array + ( id * m_sb_state_size );

            for ( int i = 0 ; i < m_sb_count && sb_id < 0 ; ++i ) {

              //  Query state of the candidate superblock.
              //  Note that the state may change at any moment
              //  as concurrent allocations and deallocations occur.
              
              const uint32_t state = *sb_state_array ;

              if ( ( ( state >> state_shift ) == block_count_lg2 ) &&
                   ( ( state &  state_mask )  <  target_count ) ) {

                //  Superblock is the required block size and
                //  its block count is below the threshold.
                sb_id = id ;
              }
              // else superblock is not usable, next superblock
              else if ( ++id < m_sb_count ) {
                sb_state_array += m_sb_state_size ;
              }
              else {
                id = 0 ;
                sb_state_array = m_sb_state_array ;
              }
            }
          }

          if ( 0 <= sb_id ) { // Found usable partfull superblock.

// printf("  found partfull( %d )\n" , sb_id );

            //  If this superblock does not match the hint
            //  and the hint is not locked then update the hint.
            //  The hint value may have changes so compare-exchange.

            if ( ( hint_sb_id != hint_lock ) &&
                 ( hint_sb_id != sb_id ) ) {
                
              Kokkos::atomic_compare_exchange
                ( hint_sb_id_ptr , hint_sb_id , sb_id );
            }
          }
          //------------------------------------------------------------------
          //  else did not find a usable partfull superblock
          //  Attempt to acquire an empty superblock.
          //  Only one thread may attempt to acquire an empty superblock
          //  for use by this block size.  Use the hint value as a lock to
          //  guarantee only one thread makes this attempt.

          else if ( ( hint_sb_id != hint_lock ) &&
                    ( hint_sb_id == Kokkos::atomic_compare_exchange
                        ( hint_sb_id_ptr , hint_sb_id , hint_lock ) ) ) {

            // This thread locked then hint

            const uint32_t claim_state = block_count_lg2 << state_shift ;

            int id = sb_id_tic ;

            sb_state_array = m_sb_state_array + ( id * m_sb_state_size );

            for ( int i = 0 ; i < m_sb_count && sb_id < 0 ; ++i ) {

              if ( ( 0u == *sb_state_array ) && 
                   ( 0u == Kokkos::atomic_compare_exchange( sb_state_array, 0u, claim_state) ) ) {
                // Superblock was empty and has been claimed
                sb_id = id ;
              }
              else {
                if ( ++id < m_sb_count ) {
                  sb_state_array += m_sb_state_size ;
                }
                else {
                  id = 0 ;
                  sb_state_array = (volatile unsigned *) m_sb_state_array ;
                }
              }
            }

// printf("  attempt empty( %d )\n" , sb_id );

            // This thread locked the hint and must unlock it.
            // If acquired a superblock then set that superblock id
            // else no empty superblock was found so set the old superblock id

            Kokkos::atomic_exchange
              ( hint_sb_id_ptr
              , uint32_t( sb_id < 0 ? hint_sb_id : sb_id ) );

            if ( sb_id < 0 ) {
              //  Failed to find an empty superblock.

              //  There is a slim chance that during all of this
              //  searching some other thread has deallocated and
              //  space became available.
              //  Iterating on this slim chance is a performance penalty
              //  with use case dependent probability of success.

              if ( 0 < retry_limit ) {
                --retry_limit ;
                continue ;
              }
              else {
                break ;
              }
            }
          }
          else if ( 0 == p ) {
            // Another thread has the lock and the allocation is not done.
            // Keep trying to obtain a superblock for the allocation.
            continue ;
          }

          //------------------------------------------------------------------
          // Arrive here with usable superblock

          if ( 0 != p ) { break ; } // Allocation previously successful

          {
            const Kokkos::pair<int,int> result =
              CB::acquire( sb_state_array , tic , block_count_lg2 );

// printf("  acquire( %d , %d )\n" , result.first , result.second );

            // If result.first < 0 then failed to acquire
            // due to either full or buffer was wrong state.
            // Could be wrong state if a deallocation raced the
            // superblock to empty before the acquire could succeed.

            if ( 0 <= result.first ) { // acquired a bit

              // Set the allocated block pointer

              p = ((char*)( m_sb_state_array + m_sb_data_offset ))
                + ( uint32_t(sb_id) << m_sb_size_lg2 ) // superblock memory
                + ( result.first    << block_size_lg2 ); // block memory

              if ( block_count_threshold != uint32_t(result.second) ) {
                break ;
              }
            }
          }

          //  Arrive here if
          //
          //  (1) Failed to allocate a block from the superblock
          //      due to racing allocations filling the superblock
          //      or racing deallocations emptying the superblock.
          //
          //  (2) Succeeded in allocating a block from the superblock
          //      and that success triggered the "help ahead" threshold
          //      where this thread attempts to update the superblock hint
          //      with a superblock that is below the threshold.
          //
          //  Repeat the superblock acquisition.

        } // end allocation attempt loop

        //--------------------------------------------------------------------
      }
      else {
        Kokkos::abort("Kokkos::Experimental::MemoryPoolv2 allocation request exceeded specified maximum allocation size");
      }

      return p ;
    }
  // end allocate
  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION
  void deallocate( void * p ) const noexcept
    {
      // Determine which superblock and block
      ptrdiff_t d = ((char*)p) -
                    ((char*)( m_sb_state_array + m_sb_data_offset ));

      const int ok_contains =
        ( 0 <= d ) && ( size_t(d) < ( m_sb_count << m_sb_size_lg2 ) );

      int ok_block_aligned = 0 ;
      int ok_dealloc_once  = 0 ;

      if ( ok_contains ) {

        const int sb_id = d >> m_sb_size_lg2 ;

        // Mask into the superblock size
        d &= ptrdiff_t( 1 << m_sb_size_lg2 ) - 1 ;

        volatile uint32_t * const sb_state_array =
          m_sb_state_array + ( sb_id * m_sb_state_size );

        const uint32_t block_count_lg2 = (*sb_state_array) >> state_shift ;
        const uint32_t block_size_lg2  = m_sb_size_lg2 - block_count_lg2 ;
        const uint32_t block_size_mask = ( 1 << block_size_lg2 ) - 1 ;

        ok_block_aligned = 0 == ( d & block_size_mask );

        if ( ok_block_aligned ) {

          const int result =
            CB::release( sb_state_array , uint32_t(d >> block_size_lg2) );

          ok_dealloc_once = 0 <= result ;

// printf("  deallocate from sb_id(%d) result(%d) bit(%d) state(0x%x)\n"
//       , sb_id
//       , result
//       , uint32_t(d >> block_size_lg2)
//       , *sb_state_array );

          if ( 0 == result ) {

            // If count at zero then change state to empty superblock

// uint32_t state =
            Kokkos::atomic_compare_exchange( sb_state_array
                                           , ( block_count_lg2 << state_shift )
                                           , 0u );

// printf("  deallocate to zero sb_id(%d) state(0x%x)\n",sb_id,state);

            // If hint was to this superblock then
            // the next allocation attempt will be unable to use
            // this superblock and will update the hint accordingly.
          }
        }
      }

      if ( ! ok_contains || ! ok_block_aligned || ! ok_dealloc_once ) {
#if 0
        printf("Kokkos::Experimental::MemoryPoolv2(0x%lx) contains(%d) block_aligned(%d) dealloc_once(%d)\n",(unsigned long)p,ok_contains,ok_block_aligned,ok_dealloc_once);
#endif
        Kokkos::abort("Kokkos::Experimental::MemoryPoolv2::deallocate given erroneous pointer");
      }
    }
  // end deallocate
  //--------------------------------------------------------------------------
};

} // namespace Experimental 
} // namespace Kokkos 

#endif /* #ifndef KOKKOS_MEMORYPOOLV2_HPP */

