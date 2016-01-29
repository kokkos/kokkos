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
#include <impl/Kokkos_Error.hpp>
#include <impl/KokkosExp_SharedAlloc.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Atomic.hpp>

// How should errors be handled?  In general, production code should return a
// value indicating failure so the user can decide how the error is handled.
// While experimental, code can abort instead.  If KOKKOS_MEMPOOL_PRINTERR is
// defined, the code will abort with an error message.  Otherwise, the code will
// return with a value indicating failure when possible, or do nothing instead.
#define KOKKOS_MEMPOOL_PRINTERR

//#define KOKKOS_MEMPOOL_PRINT_INFO

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

template < class Space >
class MemoryPool ;

namespace Impl {

#ifdef KOKKOS_MEMPOOL_PRINT_INFO
template < typename Link >
struct print_mempool {
  size_t    m_num_chunk_sizes;
  size_t *  m_chunk_size;
  Link **   m_freelist;
  char *    m_data;

  print_mempool( size_t ncs, size_t * cs, Link ** f, char * d )
    : m_num_chunk_sizes(ncs), m_chunk_size(cs), m_freelist(f), m_data(d)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()( size_t i ) const
  {
    if ( i == 0 ) {
      printf( "*** ON DEVICE ***\n");
      printf( "m_chunk_size: 0x%lx\n", m_chunk_size );
      printf( "  m_freelist: 0x%lx\n", m_freelist );
      printf( "      m_data: 0x%lx\n", m_data );
      for ( size_t l = 0; l < m_num_chunk_sizes; ++l ) {
        printf( "%2ld    freelist: 0x%lx    chunk_size: %6ld\n",
                l, m_freelist[l], m_chunk_size[l] );
      }
      printf( "                               chunk_size: %6ld\n\n",
              m_chunk_size[m_num_chunk_sizes] );
    }
  }
};
#endif

template < typename Link >
struct initialize_mempool {
  char *  m_data;
  size_t  m_chunk_size;
  size_t  m_last_chunk;

  initialize_mempool( char * d, size_t cs, size_t lc )
    : m_data(d), m_chunk_size(cs), m_last_chunk(lc)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()( size_t i ) const
  {
    Link * lp = reinterpret_cast< Link * >(m_data + i * m_chunk_size);

    // All entries in the list point to the next entry except the last which
    // is null.
    lp->m_next = i < m_last_chunk ?
                 reinterpret_cast< Link * >( m_data + (i + 1) * m_chunk_size ) :
                 reinterpret_cast< Link * >(0);
  }
};

class MemPoolList {
private:

  typedef Impl::SharedAllocationTracker  Tracker;

  template< class > friend class Kokkos::Experimental::MemoryPool;

public:

  /// \brief Structure used to create a linked list from the memory chunks.
  ///
  /// The chunks are cast to this type internally to create the lists.
  struct Link {
    Link * m_next;
  };

private:

  Tracker   m_track;

  // These three variables are pointers into device memory.
  size_t *  m_chunk_size; // Array of chunk sizes of freelists.
  Link **   m_freelist;   // Array of freelist heads.
  char *    m_data;       // Beginning memory location used for chunks.

  size_t    m_data_size;
  size_t    m_chunk_spacing;

#if defined(KOKKOS_MEMPOOL_PRINT_INFO) && defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
  mutable size_t m_count;
#endif

  ~MemPoolList() = default;
  MemPoolList() = default;
  MemPoolList( MemPoolList && ) = default;
  MemPoolList( const MemPoolList & ) = default;
  MemPoolList & operator = ( MemPoolList && ) = default;
  MemPoolList & operator = ( const MemPoolList & ) = default;

  template< class MemorySpace, class ExecutionSpace>
  inline
  MemPoolList( const MemorySpace & memspace, const ExecutionSpace &,
               size_t base_chunk_size, size_t total_size,
               size_t num_chunk_sizes, size_t chunk_spacing )
    : m_track(), m_chunk_size(0), m_freelist(0), m_data(0), m_data_size(0),
      m_chunk_spacing(chunk_spacing)
#if defined(KOKKOS_MEMPOOL_PRINT_INFO) && defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    , m_count(0)
#endif
  {
    typedef Impl::SharedAllocationRecord< MemorySpace, void >  SharedRecord;
    typedef Kokkos::RangePolicy< ExecutionSpace >              Range;

    // The base chunk size must be at least 128 bytes as this is the cache-line
    // size for NVIDA GPUs.
    if ( base_chunk_size < 128 ) {
      printf( "** Chunk size must be at least 128 bytes.  Setting to 128. **\n" );
      fflush( stdout );

      base_chunk_size = 128;
    }

    // The base chunk size must also be a multiple of 128 bytes for correct
    // memory alignment of the chunks.  If it isn't a multiple of 128, set it
    // to the smallest multiple of 128 greater than the given chunk size.
    if ( base_chunk_size % 128 != 0 ) {
      size_t old_chunk_size = base_chunk_size;
      base_chunk_size = ( ( old_chunk_size + 127 ) / 128 ) * 128;

      printf( "** Chunk size must be a multiple of 128 bytes.  Given: %ld  Using: %ld. **\n",
              old_chunk_size, base_chunk_size);
      fflush( stdout );
    }

    // Force total_size to be a multiple of base_chunk_size.
    total_size = ( ( total_size + base_chunk_size - 1 ) / base_chunk_size ) *
                 base_chunk_size;

    m_data_size = total_size;

    // Get the chunk size for the largest possible chunk.
    //   max_chunk_size =
    //     base_chunk_size * (m_chunk_spacing ^ (num_chunk_sizes - 1))
    size_t max_chunk_size = base_chunk_size;
    for (size_t i = 1; i < num_chunk_sizes; ++i) {
      max_chunk_size *= m_chunk_spacing;
    }

    // We want each chunk size to use total_size / num_chunk_sizes memory.  If
    // the total size of the pool is not enough to accomodate this, keep making
    // the next lower chunk size the max_chunk_size until it is.
    while ( max_chunk_size > total_size / num_chunk_sizes ) {
      max_chunk_size /= m_chunk_spacing;
      --num_chunk_sizes;
    }

    // We put a header at the beginnig of the device memory and use extra
    // chunks to store the header.  The header contains:
    //   size_t  chunk_size[num_chunk_sizes+1]
    //   Link *  freelist[num_chunk_sizes]

    // Calculate the size of the header where the size is rounded up to the
    // smallest multiple of base_chunk_size >= the needed size.  Assume all
    // types are 8 bytes to give ample space.
    const size_t header_bytes = ( 2 * num_chunk_sizes + 1 ) * 8;
    const size_t header_size =
      (header_bytes + base_chunk_size - 1 ) / base_chunk_size * base_chunk_size;

    // Allocate the memory including the header.
    const size_t alloc_size = total_size + header_size;

    SharedRecord * rec =
      SharedRecord::allocate( memspace, "mempool", alloc_size );

    m_track.assign_allocated_record_to_uninitialized( rec );

    // TODO: I'm not sure that sizes of types are the same on host and device.
    //       Will the pointer arithmetic hold on both host and device?  If
    //       size_t is a different size on host and device, how do I handle
    //       that?

    // Get the pointers into the allocated memory.
    char * mem = reinterpret_cast< char * >( rec->data() );
    m_chunk_size = reinterpret_cast< size_t * >( mem );
    m_freelist = reinterpret_cast< Link ** >( mem + ( num_chunk_sizes + 1 ) * 8 );
    m_data = mem + header_size;

    // Initialize the chunk sizes array.  Create num_chunk_sizes different
    // chunk sizes where each successive chunk size is
    // m_chunk_spacing * previous chunk size.  The last entry in the array is
    // 0 and is used for a stopping condition.
    m_chunk_size[0] = base_chunk_size;
    for ( size_t i = 1; i < num_chunk_sizes; ++i ) {
      m_chunk_size[i] = m_chunk_size[i - 1] * m_chunk_spacing;
    }
    m_chunk_size[num_chunk_sizes] = 0;

    size_t num_chunks[num_chunk_sizes];

    // Set the starting point in memory and get the number of chunks for each
    // freelist.  Start with the largest chunk size to ensure usage of all the
    // memory.  If there is leftover memory for a chunk size, it will be used
    // by a smaller chunk size.
    size_t used_memory = 0;
    for ( size_t i = num_chunk_sizes; i > 0; --i ) {
      // Set the starting point in the memory for the current chunk sizes's
      // freelist.
      m_freelist[i - 1] = reinterpret_cast< Link * >( m_data + used_memory );

      size_t mem_avail =
        total_size - (i - 1) * ( total_size / num_chunk_sizes ) - used_memory;

      // Set the number of chunks for the current chunk sizes's freelist.
      num_chunks[i - 1] = mem_avail / m_chunk_size[i - 1];

      used_memory += num_chunks[i - 1] * m_chunk_size[i - 1];
    }

#ifdef KOKKOS_MEMPOOL_PRINT_INFO
    printf( "\n" );
    printf( "*** ON HOST ***\n");
    printf( "m_chunk_size: 0x%lx\n", m_chunk_size );
    printf( "  m_freelist: 0x%lx\n", m_freelist );
    printf( "      m_data: 0x%lx\n", m_data );
    for ( size_t i = 0; i < num_chunk_sizes; ++i ) {
      printf( "%2ld    freelist: 0x%lx    chunk_size: %6ld    num_chunks: %8ld\n",
              i, (unsigned long) m_freelist[i], m_chunk_size[i], num_chunks[i] );
    }
    printf( "                               chunk_size: %6ld\n\n",
            m_chunk_size[num_chunk_sizes] );
    fflush( stdout );
#endif

#ifdef KOKKOS_MEMPOOL_PRINTERR
    if ( used_memory != total_size ) {
      printf( "\n** MemoryPool::MemoryPool() USED_MEMORY(%ld) != TOTAL_SIZE(%ld) **\n",
              used_memory, total_size );
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      fflush( stdout );
#endif
      Kokkos::abort( "" );
    }
#endif

    // Create the chunks for each freelist.
    for ( size_t i = 0; i < num_chunk_sizes; ++i ) {
      // Initialize the next pointers to point to the next chunk for all but the
      // last chunk which has points to NULL.
      initialize_mempool< Link >
        im( (char *) m_freelist[i], m_chunk_size[i], num_chunks[i] - 1 );
      Kokkos::Impl::ParallelFor< initialize_mempool< Link >, Range >
        closure( im, Range( 0, num_chunks[i] ) );

      closure.execute();

      ExecutionSpace::fence();
    }

#ifdef KOKKOS_MEMPOOL_PRINT_INFO
    print_mempool < Link > pm( num_chunk_sizes, m_chunk_size, m_freelist, m_data );

    Kokkos::Impl::ParallelFor< print_mempool< Link >, Range >
      closure( pm, Range( 0, 10 ) );

    closure.execute();

    ExecutionSpace::fence();
#endif
  }

  ///\brief  Inserts an already linked list of chunks into the pool.
  KOKKOS_FUNCTION
  void insert_list( Link * lp_head, Link * lp_tail, size_t list ) const;

  ///\brief  Claim chunks of untracked memory from the pool.
  KOKKOS_FUNCTION
  void * allocate( size_t alloc_size ) const;

  ///\brief  Release claimed memory back into the pool.
  KOKKOS_FUNCTION
  void deallocate( void * alloc_ptr, size_t alloc_size ) const;

  // The following three functions are used for debugging.
  void print_status() const
  {
    for ( size_t l = 0; m_chunk_size[l] > 0; ++l ) {
      size_t count = 0;
      Link * chunk = m_freelist[l];

      while ( chunk != NULL ) {
        ++count;
        chunk = chunk->m_next;
      }

      printf( "chunk_size: %6ld    num_chunks: %8ld\n", m_chunk_size[l], count );
    }
  }

  size_t get_min_chunk_size() const { return m_chunk_size[0]; }
  size_t get_mem_size() const { return m_data_size; }
};

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
/*  Prefer to implement these functions in a separate
 *  compilation unit.  However, the 'nvcc' linker
 *  has an internal error when attempting separate compilation
 *  (--relocatable-device-code=true)
 *  of Kokkos unit tests.
 */
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA )

#include <impl/Kokkos_MemoryPool_Inline.hpp>

#endif

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

/// \class MemoryPool
/// \brief Memory management for pool of same-sized chunks of memory.
///
/// MemoryPool is a memory space that can be on host or device.  It provides a
/// pool memory allocator for fast allocation of same-sized chunks of memory.
/// The memory is only accessible on the host / device this allocator is
/// associated with.
template < class Space >
class MemoryPool {
private:

  Impl::MemPoolList  m_memory;

  typedef typename Space::memory_space     backend_memory_space;
  typedef typename Space::execution_space  execution_space;

public:

  //! Tag this class as a kokkos memory space
  typedef MemoryPool  memory_space;

  //------------------------------------

  MemoryPool() = default;
  MemoryPool( MemoryPool && rhs ) = default;
  MemoryPool( const MemoryPool & rhs ) = default;
  MemoryPool & operator = ( MemoryPool && ) = default;
  MemoryPool & operator = ( const MemoryPool & ) = default;
  ~MemoryPool() = default;

  /// \brief  Allocate memory pool
  /// \param memspace         From where to allocate the pool.
  /// \param base_chunk_size  Hand out memory in chunks of this size.
  /// \param total_size       Total size of the pool.
  MemoryPool( const backend_memory_space & memspace,
              size_t base_chunk_size, size_t total_size,
              size_t num_chunk_sizes = 4, size_t chunk_spacing = 4 )
    : m_memory( memspace, execution_space(), base_chunk_size, total_size,
                num_chunk_sizes, chunk_spacing )
  {}

  KOKKOS_INLINE_FUNCTION
  bool is_empty() const { return 0 == *m_freelist.m_head_list ; }

  KOKKOS_INLINE_FUNCTION
  unsigned chunk_size() const { return m_freelist.m_chunk_size ; }

  ///\brief  Claim chunks of untracked memory from the pool.
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void * allocate( const size_t alloc_size ) const
  { return m_memory.allocate( alloc_size ); }

  ///\brief  Release claimed memory back into the pool
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void deallocate( void * const alloc_ptr, const size_t alloc_size ) const
  { m_memory.deallocate( alloc_ptr, alloc_size ); }

  // The following three functions are used for debugging.
  void print_status() const { m_memory.print_status(); }
  size_t get_min_chunk_size() const { return m_memory.get_min_chunk_size(); }
  size_t get_mem_size() const { return m_memory.get_mem_size(); }
};

} // namespace Experimental
} // namespace Kokkos

#ifdef KOKKOS_MEMPOOL_PRINTERR
#undef KOKKOS_MEMPOOL_PRINTERR
#endif

#ifdef KOKKOS_MEMPOOL_PRINT_INFO
#undef KOKKOS_MEMPOOL_PRINT_INFO
#endif

#endif /* #define KOKKOS_MEMORYPOOL_HPP */
