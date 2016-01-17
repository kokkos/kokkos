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

#include <iostream>

// TODO: This should probably not be included here, but it's the only way I
//       could get this to compile.  Ask Carter about this.
#include <Kokkos_Core.hpp>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_MemoryTraits.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_Timer.hpp>

//#define MEMPOOL_SERIAL
//#define MEMPOOL_PRINTERR

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

// Global variable for this file only.
#define MEMPOOL_LIST_LOCK (Link*)0xFFFFFFFFFFFFFFFF

namespace Impl {

template < typename ExecutionSpace, typename Link >
struct initialize_mempool {
  typedef ExecutionSpace  execution_space;
  typedef typename execution_space::size_type    size_type;

  size_type m_chunk_size;
  char * m_head;

  initialize_mempool( size_type num_chunks, size_type cs, char * h )
    : m_chunk_size( cs ), m_head( h )
  {
    // Initialize the view with the out degree of each vertex.
    Kokkos::parallel_for( num_chunks, *this );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( size_type i ) const
  {
    Link * lp = reinterpret_cast< Link * >( m_head + i * m_chunk_size );
    lp->m_next = 0;
  }
};

template < typename ExecutionSpace, typename Link >
struct initialize_mempool2 {
  typedef ExecutionSpace  execution_space;
  typedef typename execution_space::size_type    size_type;

  size_type m_chunk_size;
  char * m_head;

  initialize_mempool2( size_type num_chunks, size_type cs, char * h )
    : m_chunk_size( cs ), m_head( h )
  {
    // Initialize the view with the out degree of each vertex.
    Kokkos::parallel_for( num_chunks, *this );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( size_type i ) const
  {
    Link * lp = reinterpret_cast< Link * >( m_head + i * m_chunk_size );
    lp->m_next = reinterpret_cast< Link * >( m_head + ( i + 1 ) * m_chunk_size );
  }
};

} // namespace Impl

/// \class MemoryPool
/// \brief Memory management for pool of same-sized chunks of memory.
///
/// MemoryPool is a memory space that can be on host or device.  It provides a
/// pool memory allocator for fast allocation of same-sized chunks of memory.
/// The memory is only accessible on the host / device this allocator is
/// associated with.
template < class MemorySpace >
class MemoryPool {
public:
  /// \brief Structure used to create a linked list from the memory chunk.
  ///
  /// The allocated memory is cast to this type internally to create the lists.
  struct Link {
    Link * m_next;
  };

  //! Tag this class as a kokkos memory space
  typedef MemoryPool                                         memory_space;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
  typedef typename MemorySpace::execution_space execution_space;

  //! The memory space's preferred device_type.
  typedef typename MemorySpace::device_type device_type;

  typedef typename MemorySpace::size_type                    size_type;
  typedef Impl::SharedAllocationRecord< MemorySpace, void >  SharedRecord;
  typedef Impl::SharedAllocationTracker                      Tracker;
  typedef Kokkos::View< Link, execution_space >              LinkView;

  //------------------------------------

  MemoryPool() = default;
  MemoryPool( MemoryPool && rhs ) = default;
  MemoryPool( const MemoryPool & rhs ) = default;
  MemoryPool & operator = ( MemoryPool && ) = default;
  MemoryPool & operator = ( const MemoryPool & ) = default;
  ~MemoryPool() = default;

  /**\brief  Default memory space instance. */
  MemoryPool( const MemorySpace & memspace, /* From where to allocate the pool. */
              size_type chunk_size, /* Hand out memory in chunks of this size. */
              size_type total_size /* Total size of the pool. */
            )
    : m_freelist( "freelist" )
  {
    // TODO: Ask Carter about how to better handle this.
    if ( chunk_size < 8 ) {
#ifdef MEMPOOL_PRINTERR
      fprintf( stderr, "Chunk size must be at least 8 bytes.  Setting to be 8 bytes.\n" );
      fflush( stderr );
#endif

      chunk_size = 8;
    }

    m_chunk_size = chunk_size;

    // Force total_size to be a multiple of chunk_size.
    total_size = ( ( total_size + chunk_size - 1 ) / chunk_size ) * chunk_size;

    size_type num_chunks = total_size / chunk_size;

    SharedRecord * rec = SharedRecord::allocate( memspace, "mempool", total_size );
    m_pool_tracker.assign_allocated_record_to_uninitialized( rec );

    Link alink;
    alink.m_next = static_cast< Link * >( rec->data() );
    deep_copy(m_freelist, alink);

    // Initialize all next pointers to 0.  This is a bit overkill since only the
    // next pointer of the last chunk needs to be set to 0, but two simple loops
    // should be faster than one loop with an if statement.
    {
      Impl::initialize_mempool< execution_space, Link >
        im( num_chunks, m_chunk_size, static_cast< char * >( rec->data() ) );
    }

    // Initialize the next pointers to point to the next chunk for all but the
    // last chunk.
    {
      Impl::initialize_mempool2< execution_space, Link >
        im( num_chunks - 1, m_chunk_size, static_cast< char * >( rec->data() ) );
    }
  }

  ///\brief  Claim chunks of untracked memory from the pool.
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void * allocate( const size_t alloc_size ) const
  {
#ifdef MEMPOOL_PRINTERR
    if ( alloc_size > m_chunk_size ) {
      // This is just here for now for debugging as we only support allocating
      // m_chunk_size or smaller.
      fprintf( stderr, "MemoryPool::allocate() ALLOC_SIZE(%ld) > CHUNK_SIZE(%ld)\n",
               alloc_size, m_chunk_size );
      fflush(stderr);
    }
#endif

#ifdef MEMPOOL_SERIAL
    void * p = static_cast< void * >( m_freelist().m_next );
    m_freelist().m_next = m_freelist().m_next->m_next;
    return p;
#else
    Link * volatile * freelist = &(m_freelist().m_next);
    void * p = 0;

    bool removed = false;

    while ( ! removed ) {
      Link * const old_head = *freelist;

      if ( old_head == 0 ) {
        // The freelist is empty.  Just return 0.
        removed = true;

#ifdef MEMPOOL_PRINTERR
        //  TODO: Should I throw an error when the pool is out of memory?  For
        //        now, I will just print an error and return 0.
        fprintf( stderr, "MemoryPool::allocate() OUT_OF_MEMORY\n" );
        fflush(stderr);
#endif
      }
      else if ( old_head != MEMPOOL_LIST_LOCK ) {
        // In the initial look at the head, the freelist wasn't empty or
        // locked. Attempt to lock the head of list.  If the list was changed
        // (including being locked) between the initial look and now, head will
        // be different than old_head.  This means the removal can't proceed
        // and has to be tried again.
        Link * const head = atomic_compare_exchange( freelist, old_head, MEMPOOL_LIST_LOCK );

        if ( head == old_head ) {
          // The lock succeeded.  Get a local copy of the second entry in
          // the list.
          Link * const head_next = *((Link * volatile *) &(old_head->m_next) );

          // Replace the lock with the next entry on the list.
          Link * const l = atomic_compare_exchange( freelist, MEMPOOL_LIST_LOCK, head_next );

          if ( l != MEMPOOL_LIST_LOCK ) {
#ifdef MEMPOOL_PRINTERR
            // We shouldn't get here, but this check is here for sanity.  
            fprintf( stderr, "MemoryPool::allocate() UNLOCK_ERROR(0x%lx)\n",
                     (unsigned long) freelist );
            fflush(stderr);
#endif
          }

          *((Link * volatile *) &(old_head->m_next) ) = 0 ;
          p = old_head;
          removed = true;
        }
      }
      else {
        // The freelist was locked.  For now, the thread will just do the next
        // loop iteration.  If this is a performance issue, it can be revisited.
      }
    }

    return p;
#endif
  }

  ///\brief  Release claimed memory back into the pool
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void deallocate( void * const alloc_ptr,
                   const size_t alloc_size ) const
  {
#ifdef MEMPOOL_SERIAL
    Link * lp = static_cast< Link * >( alloc_ptr );
    lp->m_next = m_freelist().m_next;
    m_freelist().m_next = lp;
#else
    Link * lp = static_cast< Link * >( alloc_ptr );
    Link * volatile * freelist = &(m_freelist().m_next);

    bool inserted = false;

    while ( ! inserted ) {
      Link * const old_head = *freelist;

      if ( old_head != MEMPOOL_LIST_LOCK ) {
        // In the initial look at the head, the freelist wasn't locked.

        // Proactively assign lp->m_next assuming a successful insertion into
        // the list.
        *((Link * volatile *) &(lp->m_next)) = old_head;

        memory_fence();

        // Attempt to insert at head of list.  If the list was changed
        // (including being locked) between the initial look and now, head will
        // be different than old_head.  This means the insert can't proceed and
        // has to be tried again.
        Link * const head = atomic_compare_exchange( freelist, old_head, lp );

        if ( head == old_head ) {
          inserted = true;
        }
      }
    }
#endif
  }

private:
  Tracker    m_pool_tracker;
  LinkView   m_freelist;
  size_type  m_chunk_size;
};

} // namespace Experimental
} // namespace Kokkos

#ifdef MEMPOOL_SERIAL
#undef MEMPOOL_SERIAL
#endif

#ifdef MEMPOOL_LIST_LOCK
#undef MEMPOOL_LIST_LOCK
#endif

#endif /* #define KOKKOS_MEMORYPOOL_HPP */
