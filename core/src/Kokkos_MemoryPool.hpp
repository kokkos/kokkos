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

#ifndef KOKKOS_MEMORYPOOL_HPP
#define KOKKOS_MEMORYPOOL_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_Atomic.hpp>
#include <impl/Kokkos_ConcurrentBitset.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {

template< typename DeviceType >
class MemoryPool {
private:

  typedef typename Kokkos::Impl::concurrent_bitset CB ;

  enum : uint32_t { bits_per_int_lg2  = CB::bits_per_int_lg2 };
  enum : uint32_t { state_shift       = CB::state_shift };
  enum : uint32_t { state_used_mask   = CB::state_used_mask };
  enum : uint32_t { state_header_mask = CB::state_header_mask };
  enum : uint32_t { max_bit_count_lg2 = CB::max_bit_count_lg2 };
  enum : uint32_t { max_bit_count     = CB::max_bit_count };

  /*  Defaults for min block, max block, and superblock sizes */
  enum : uint32_t { MIN_BLOCK_SIZE_LG2  =  6  /*   64 bytes */ };
  enum : uint32_t { MAX_BLOCK_SIZE_LG2  = 12  /*   4k bytes */ };
  enum : uint32_t { SUPERBLOCK_SIZE_LG2 = 16  /*  64k bytes */ };

  enum : uint32_t { HINT_PER_BLOCK_SIZE = 2 };

  /*  Each superblock has a concurrent bitset state
   *  which is an array of uint32_t integers.
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
  uint32_t   m_min_block_size_lg2 ;
  uint32_t   m_max_block_size_lg2 ;
  uint32_t   m_hint_offset ;   // Offset to K * #block_size array of hints
  uint32_t   m_data_offset ;   // Offset to 0th superblock data
  uint32_t   m_padding ;

public:

  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION
  size_t capacity() const noexcept
    { return ( 1LU << m_sb_size_lg2 ) * m_sb_count ; }

  KOKKOS_INLINE_FUNCTION
  size_t min_block_size() const noexcept
    { return ( 1LU << m_min_block_size_lg2 ); }

  KOKKOS_INLINE_FUNCTION
  size_t max_block_size() const noexcept
    { return ( 1LU << m_max_block_size_lg2 ); }

  struct usage_statistics {
    size_t capacity_bytes ;       ///<  Capacity in bytes
    size_t superblock_bytes ;     ///<  Superblock size in bytes
    size_t max_block_bytes ;      ///<  Maximum block size in bytes
    size_t min_block_bytes ;      ///<  Minimum block size in bytes
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

      const size_t alloc_size = m_hint_offset * sizeof(uint32_t);

      uint32_t * const sb_state_array = 
        accessible ? m_sb_state_array : (uint32_t *) host.allocate(alloc_size);

      if ( ! accessible ) {
        Kokkos::Impl::DeepCopy< Kokkos::HostSpace , base_memory_space >
          ( sb_state_array , m_sb_state_array , alloc_size );
      }

      stats.superblock_bytes = ( 1LU << m_sb_size_lg2 );
      stats.max_block_bytes  = ( 1LU << m_max_block_size_lg2 );
      stats.min_block_bytes  = ( 1LU << m_min_block_size_lg2 );
      stats.capacity_bytes   = stats.superblock_bytes * m_sb_count ;
      stats.capacity_superblocks = m_sb_count ;
      stats.consumed_superblocks = 0 ;
      stats.consumed_blocks = 0 ;
      stats.consumed_bytes  = 0 ;
      stats.reserved_blocks = 0 ;
      stats.reserved_bytes  = 0 ;

      const uint32_t * sb_state_ptr = sb_state_array ;

      for ( uint32_t i = 0 ; i < m_sb_count
          ; ++i , sb_state_ptr += m_sb_state_size ) {

        const uint32_t block_count_lg2 = (*sb_state_ptr) >> state_shift ;

        if ( block_count_lg2 ) {
          const uint32_t block_count    = 1u << block_count_lg2 ;
          const uint32_t block_size_lg2 = m_sb_size_lg2 - block_count_lg2 ;
          const uint32_t block_size     = 1u << block_size_lg2 ;
          const uint32_t block_used     = (*sb_state_ptr) & state_used_mask ;

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

  void print_state( std::ostream & s ) const
    {
      Kokkos::HostSpace host ;

      const size_t alloc_size = m_hint_offset * sizeof(uint32_t);

      uint32_t * const sb_state_array = 
        accessible ? m_sb_state_array : (uint32_t *) host.allocate(alloc_size);

      if ( ! accessible ) {
        Kokkos::Impl::DeepCopy< Kokkos::HostSpace , base_memory_space >
          ( sb_state_array , m_sb_state_array , alloc_size );
      }

      const uint32_t * sb_state_ptr = sb_state_array ;

      s << "pool_size(" << ( ( 1 << m_sb_size_lg2 ) * m_sb_count ) << ")"
        << " superblock_size(" << ( 1 << m_sb_size_lg2 ) << ")" << std::endl ;

      for ( uint32_t i = 0 ; i < m_sb_count
          ; ++i , sb_state_ptr += m_sb_state_size ) {

        if ( *sb_state_ptr ) {

          const uint32_t block_count_lg2 = (*sb_state_ptr) >> state_shift ;
          const uint32_t block_size_lg2  = m_sb_size_lg2 - block_count_lg2 ;
          const uint32_t block_count     = 1 << block_count_lg2 ;
          const uint32_t block_used      = (*sb_state_ptr) & state_used_mask ;

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

  //--------------------------------------------------------------------------

  MemoryPool() = default ;
  MemoryPool( MemoryPool && ) = default ;
  MemoryPool( const MemoryPool & ) = default ;
  MemoryPool & operator = ( MemoryPool && ) = default ;
  MemoryPool & operator = ( const MemoryPool & ) = default ;

  /**\brief  Allocate a memory pool from 'memspace'.
   *
   *  The memory pool will have at least 'min_total_alloc_size' bytes
   *  of memory to allocate divided among superblocks of at least
   *  'min_superblock_size' bytes.  A single allocation must fit
   *  within a single superblock, so 'min_superblock_size' must be
   *  at least as large as the maximum single allocation.
   *  Both 'min_total_alloc_size' and 'min_superblock_size'
   *  are rounded up to the smallest power-of-two value that
   *  contains the corresponding sizes.
   *  Individual allocations will always consume a block of memory that
   *  is also a power-of-two.  These roundings are made to enable
   *  significant runtime performance improvements.
   */
  MemoryPool( const base_memory_space & memspace
            , const size_t   min_total_alloc_size
            , const uint32_t min_block_alloc_size // = 1 << MIN_BLOCK_SIZE_LG2
            , const uint32_t max_block_alloc_size // = 1 << MAX_BLOCK_SIZE_LG2
            , const uint32_t min_superblock_size  // = 1 << SUPERBLOCK_SIZE_LG2
            )
    : m_tracker()
    , m_sb_state_array(0)
    , m_sb_state_size(0)
    , m_sb_size_lg2(0)
    , m_sb_count(0)
    , m_min_block_size_lg2(0)
    , m_max_block_size_lg2(0)
    , m_hint_offset(0)
    , m_data_offset(0)
    {
      const uint32_t int_align_lg2  = 3 ; /* align as int[8] */
      const uint32_t int_align_mask = ( 1u << int_align_lg2 ) - 1 ;

      // Block and superblock size is power of two:

      m_min_block_size_lg2 =
        Kokkos::Impl::integral_power_of_two_that_contains(min_block_alloc_size);

      m_max_block_size_lg2 =
        Kokkos::Impl::integral_power_of_two_that_contains(max_block_alloc_size);
  
      m_sb_size_lg2 =
        Kokkos::Impl::integral_power_of_two_that_contains(min_superblock_size);

      // Constraints:
      // m_min_block_size_lg2 <= m_max_block_size_lg2 <= m_sb_size_lg2
      // m_sb_size_lg2 <= m_min_block_size + max_bit_count_lg2

      if ( m_min_block_size_lg2 + max_bit_count_lg2 < m_sb_size_lg2 ) {
        m_min_block_size_lg2 = m_sb_size_lg2 - max_bit_count_lg2 ;
      }
      if ( m_min_block_size_lg2 + max_bit_count_lg2 < m_max_block_size_lg2 ) {
        m_min_block_size_lg2 = m_max_block_size_lg2 - max_bit_count_lg2 ;
      }
      if ( m_max_block_size_lg2 < m_min_block_size_lg2 ) {
        m_max_block_size_lg2 = m_min_block_size_lg2 ;
      }
      if ( m_sb_size_lg2 < m_max_block_size_lg2 ) {
        m_sb_size_lg2 = m_max_block_size_lg2 ;
      }

      // At least 32 minimum size blocks in a superblock

      if ( m_sb_size_lg2 < m_min_block_size_lg2 + 5 ) {
        m_sb_size_lg2 = m_min_block_size_lg2 + 5 ;
      }

      // number of superblocks is multiple of superblock size that
      // can hold min_total_alloc_size.

      const uint32_t sb_size_mask = ( 1u << m_sb_size_lg2 ) - 1 ;

      m_sb_count = ( min_total_alloc_size + sb_size_mask ) >> m_sb_size_lg2 ;

      // Any superblock can be assigned to the smallest size block
      // Size the block bitset to maximum number of blocks

      const uint32_t max_block_count_lg2 =
        m_sb_size_lg2 - m_min_block_size_lg2 ;

      m_sb_state_size =
        ( CB::buffer_bound_lg2( max_block_count_lg2 ) + int_align_mask ) & ~int_align_mask ;

      // Array of all superblock states

      const size_t all_sb_state_size =
        ( m_sb_count * m_sb_state_size + int_align_mask ) & ~int_align_mask ;

      // Number of block sizes

      const uint32_t number_block_sizes =
         1 + m_max_block_size_lg2 - m_min_block_size_lg2 ;

      // Array length for possible block sizes
      // Hint array is one uint32_t per block size

      const uint32_t block_size_array_size =
        ( number_block_sizes + int_align_mask ) & ~int_align_mask ;

      m_hint_offset = all_sb_state_size ;
      m_data_offset = m_hint_offset +
                      block_size_array_size * HINT_PER_BLOCK_SIZE ;

      // Allocation:

      const size_t header_size = m_data_offset * sizeof(uint32_t);
      const size_t alloc_size  = header_size + ( m_sb_count << m_sb_size_lg2 );

      Record * rec = Record::allocate( memspace , "MemoryPool" , alloc_size );

      m_tracker.assign_allocated_record_to_uninitialized( rec );

      m_sb_state_array = (uint32_t *) rec->data();

      Kokkos::HostSpace host ;

      uint32_t * const sb_state_array = 
        accessible ? m_sb_state_array
                   : (uint32_t *) host.allocate(header_size);

      for ( uint32_t i = 0 ; i < m_data_offset ; ++i ) sb_state_array[i] = 0 ;

      // Initial assignment of empty superblocks to block sizes:

      for ( uint32_t i = 0 , j = 0 ; i < number_block_sizes ; ++i ) {
        const uint32_t block_size_lg2  = i + m_min_block_size_lg2 ;
        const uint32_t block_count_lg2 = m_sb_size_lg2 - block_size_lg2 ;
        const uint32_t block_state     = block_count_lg2 << state_shift ;
        const uint32_t hint_begin = m_hint_offset + i * HINT_PER_BLOCK_SIZE ;

        // for block size index 'i':
        //   sb_id_hint = sb_state_array[ hint_begin ];
        //   sb_id_end  = sb_state_array[ hint_begin + 1 ];

        sb_state_array[ hint_begin ] = j ;
        sb_state_array[ hint_begin + 1 ] = j ;

        const uint32_t kbeg = ( i * m_sb_count ) / number_block_sizes ;
        const uint32_t kend = ( ( i + 1 ) * m_sb_count ) / number_block_sizes ;

        for ( uint32_t k = kbeg ; k < kend ; ++k , ++j ) {
          sb_state_array[ j * m_sb_state_size ] = block_state ;
        }
      }

      if ( ! accessible ) {
        Kokkos::Impl::DeepCopy< base_memory_space , Kokkos::HostSpace >
          ( m_sb_state_array , sb_state_array , header_size );

        host.deallocate( sb_state_array, header_size );
      }
    }

  //--------------------------------------------------------------------------

private:

  /* Given a size 'n' get the block size in which it can be allocated.
   * Restrict lower bound to minimum block size.
   */
  KOKKOS_FORCEINLINE_FUNCTION
  unsigned get_block_size_lg2( unsigned n ) const noexcept
    {
      const unsigned i = Kokkos::Impl::integral_power_of_two_that_contains( n );

      return i < m_min_block_size_lg2 ? m_min_block_size_lg2 : i ;
    }

public:

  KOKKOS_INLINE_FUNCTION
  uint32_t allocate_block_size( uint32_t alloc_size ) const noexcept
    {
      return alloc_size <= (1UL << m_max_block_size_lg2)
           ? ( 1u << get_block_size_lg2( alloc_size ) )
           : 0 ;
    }

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
      void * p = 0 ;

      if ( alloc_size <= (1UL << m_max_block_size_lg2) ) {

        // Allocation will fit within a superblock
        // that has block sizes ( 1 << block_size_lg2 )

        const uint32_t block_size_lg2   = get_block_size_lg2( alloc_size );
        const uint32_t block_count_lg2  = m_sb_size_lg2 - block_size_lg2 ;
        const uint32_t block_count      = 1u << block_count_lg2 ;
        const uint32_t block_state      = block_count_lg2 << state_shift ;
        const uint32_t block_count_mask = block_count - 1 ;

        // hint_sb_id_ptr[0] is the dynamically changing hint
        // hint_sb_id_ptr[1] is the designated start point for this block size

        volatile uint32_t * const hint_sb_id_ptr =
          m_sb_state_array + m_hint_offset +
          ( block_size_lg2 - m_min_block_size_lg2 ) * HINT_PER_BLOCK_SIZE ;

        const uint32_t block_size_sb_id_begin = hint_sb_id_ptr[1] ;

        // Fast query clock register 'tic' to pseudo-randomize
        // the guess for which block within a superblock should
        // be claimed.  If not available then a search occurs.

        const uint32_t block_id_hint = block_count_mask &
          (uint32_t)( Kokkos::Impl::clock_tic()
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA )
          // Spread out potentially concurrent access
          // by threads within a warp or thread block.
          + ( threadIdx.x + blockDim.x * threadIdx.y )
#endif
          );

        int sb_id = -1 ;

        volatile uint32_t * sb_state_array = 0 ;

        while(1) {

          uint32_t hint_sb_id = 0 ;

          if ( sb_id < 0 ) {

            hint_sb_id = *hint_sb_id_ptr ;

            sb_id = hint_sb_id ;

            sb_state_array = m_sb_state_array + m_sb_state_size * sb_id ;
          }

          if ( 0 <= sb_id ) {

            const Kokkos::pair<int,int> result =
              CB::acquire_bounded_lg2( sb_state_array
                                     , block_count_lg2 /* count */
                                     , block_id_hint
                                     , block_state
                                     );

// printf("  acquire block_count_lg2(%d) block_state(0x%x) sb_id(%d) bit(%d)\n" , block_count_lg2 , block_state , sb_id , result.first );

            // If result.first < 0 then failed to acquire
            // due to either full or buffer was wrong state.
            // Could be wrong state if a deallocation raced the
            // superblock to empty before the acquire could succeed.

            if ( 0 <= result.first ) { // acquired a bit

              // Set the allocated block pointer

              p = ((char*)( m_sb_state_array + m_data_offset ))
                + ( uint32_t(sb_id) << m_sb_size_lg2 ) // superblock memory
                + ( result.first    << block_size_lg2 ); // block memory

              break ; // Success
            }
          }
          //------------------------------------------------------------------
          //  Arrive here if failed to acquire a block.
          //  Must find a new superblock.

          //  Start searching at designated index for this block size.
          //  Look for a partially full superblock of this block size.
          //  Look for an empty superblock just in case cannot find partfull.

          sb_id = -1 ;

          bool update_hint = false ;

          uint32_t state_empty = 0 ;

          int sb_id_empty = -1 ;

          sb_state_array =
            m_sb_state_array + block_size_sb_id_begin * m_sb_state_size ;

          for ( uint32_t i = 0 , id = block_size_sb_id_begin ;
                i < m_sb_count && sb_id < 0 ; ++i ) {

            //  Query state of the candidate superblock.
            //  Note that the state may change at any moment
            //  as concurrent allocations and deallocations occur.
            
            const uint32_t state = *sb_state_array ;
            const uint32_t used  = state & state_used_mask ;

            if ( ( ( state & state_header_mask ) == block_state ) &&
                 ( used < block_count ) ) {
              //  Superblock is the required block size and
              //  its block count is below the threshold.
              sb_id = id ;

              update_hint = used + 1 < block_count ;
            }
            else {

              if ( ( used == 0 ) && ( sb_id_empty == -1 ) ) {
                // The first empty superblock following the
                // designated superblock, save to use if a
                // partfull superblock is not found.
                sb_id_empty = id ;
                state_empty = state ;
              }

              if ( ++id < m_sb_count ) {
                sb_state_array += m_sb_state_size ;
              }
              else {
                id = 0 ;
                sb_state_array = m_sb_state_array ;
              }
            }
          }

// printf("  search m_sb_count(%d) sb_id(%d) sb_id_empty(%d)\n" , m_sb_count , sb_id , sb_id_empty );

          if ( sb_id < 0 ) {
            if ( 0 <= sb_id_empty ) {

              //  Did not find a partfull superblock
              //  Found first empty superblock following designated superblock
              //  Attempt to claim it for this block size.
              //  If the claim fails assume that another thread claimed it
              //  for this block size and try to use it anyway,
              //  but do not update hint.

              sb_id = sb_id_empty ;

              sb_state_array = m_sb_state_array + ( sb_id * m_sb_state_size );

              //  If successfully changed assignment of empty superblock 'sb_id'
              //  to this block_size then update the hint.

              update_hint = state_empty ==
                Kokkos::atomic_compare_exchange
                  (sb_state_array,state_empty,block_state);
            }
            else if ( 0 < retry_limit ) {
              //  Failed to find a partfull superblock.
              //  Failed to find an empty superblock.

              //  There is a slim chance that during all of this
              //  searching some other thread has deallocated and
              //  space became available.
              //  Iterating on this slim chance is a performance penalty
              //  with use case dependent probability of success.

              --retry_limit ;
            }
            else {
              //  Failed to find a partfull superblock.
              //  Failed to find an empty superblock.
              //  Retries have exhausted.
              //  Allocation fails.

              break ;
            }
          }

          if ( update_hint && ( int(hint_sb_id) != sb_id ) ) {

// printf("  update hint block_size_lg2(%d) sb_id(%d)\n" , block_size_lg2 , sb_id );

            Kokkos::atomic_compare_exchange
                ( hint_sb_id_ptr , hint_sb_id , sb_id );
          }
        } // end allocation attempt loop

        //--------------------------------------------------------------------
      }
      else {
        Kokkos::abort("Kokkos MemoryPool allocation request exceeded specified maximum allocation size");
      }

      return p ;
    }
  // end allocate
  //--------------------------------------------------------------------------

  /**\brief  Return an allocated block of memory to the pool.
   *
   *  Requires: p is return value from allocate( alloc_size );
   *
   *  For now the alloc_size is ignored.
   */
  KOKKOS_INLINE_FUNCTION
  void deallocate( void * p , size_t /* alloc_size */ ) const noexcept
    {
      // Determine which superblock and block
      const ptrdiff_t d =
        ((char*)p) - ((char*)( m_sb_state_array + m_data_offset ));

      // Verify contained within the memory pool's superblocks:
      const int ok_contains =
        ( 0 <= d ) && ( size_t(d) < ( m_sb_count << m_sb_size_lg2 ) );

      int ok_block_aligned = 0 ;
      int ok_dealloc_once  = 0 ;

      if ( ok_contains ) {

        const int sb_id = d >> m_sb_size_lg2 ;

        // State array for the superblock.
        volatile uint32_t * const sb_state_array =
          m_sb_state_array + ( sb_id * m_sb_state_size );

        const uint32_t block_state    = (*sb_state_array) & state_header_mask ;
        const uint32_t block_size_lg2 =
          m_sb_size_lg2 - ( block_state >> state_shift );

        ok_block_aligned = 0 == ( d & ( ( 1 << block_size_lg2 ) - 1 ) );

        if ( ok_block_aligned ) {

          // Map address to block's bit
          // mask into superblock and then shift down for block index

          const uint32_t bit =
            ( d & ( ptrdiff_t( 1 << m_sb_size_lg2 ) - 1 ) ) >> block_size_lg2 ;

          const int result =
            CB::release( sb_state_array , bit , block_state );

          ok_dealloc_once = 0 <= result ;

// printf("  deallocate from sb_id(%d) result(%d) bit(%d) state(0x%x)\n"
//       , sb_id
//       , result
//       , uint32_t(d >> block_size_lg2)
//       , *sb_state_array );

        }
      }

      if ( ! ok_contains || ! ok_block_aligned || ! ok_dealloc_once ) {
#if 1
        printf("Kokkos MemoryPool deallocate(0x%lx) contains(%d) block_aligned(%d) dealloc_once(%d)\n",(uintptr_t)p,ok_contains,ok_block_aligned,ok_dealloc_once);
#endif
        Kokkos::abort("Kokkos MemoryPool::deallocate given erroneous pointer");
      }
    }
  // end deallocate
  //--------------------------------------------------------------------------
};

} // namespace Kokkos 

#endif /* #ifndef KOKKOS_MEMORYPOOL_HPP */

