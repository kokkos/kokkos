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

#ifndef KOKKOS_MEMORYPOOL_CPP
#define KOKKOS_MEMORYPOOL_CPP

#define KOKKOS_MEMPOOLLIST_LOCK reinterpret_cast<Link *>( ~uintptr_t(0) )

// How should errors be handled?  In general, production code should return a
// value indicating failure so the user can decide how the error is handled.
// While experimental, code can abort instead.  If KOKKOS_MEMPOOLLIST_PRINTERR
// is defined, the code will abort with an error message.  Otherwise, the code
// will return with a value indicating failure when possible, or do nothing
// instead.
//#define KOKKOS_MEMPOOLLIST_PRINTERR

//#define KOKKOS_MEMPOOLLIST_PRINT_INFO

//----------------------------------------------------------------------------

#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA )

/* This '.cpp' is being included by the header file
 * to inline these functions for Cuda.
 *
 *  Prefer to implement these functions in a separate
 *  compilation unit.  However, the 'nvcc' linker
 *  has an internal error when attempting separate compilation
 *  (--relocatable-device-code=true)
 *  of Kokkos unit tests.
 */

#define KOKKOS_MEMPOOLLIST_INLINE inline

#else

/*  This '.cpp' file is being separately compiled for the Host */

#include <Kokkos_MemoryPool.hpp>
#include <Kokkos_Atomic.hpp>

#define KOKKOS_MEMPOOLLIST_INLINE /* */

#endif

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

#if defined(KOKKOS_MEMPOOLLIST_PRINT_INFO) && defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
long MemPoolList::m_count = 0;
#endif

KOKKOS_FUNCTION
KOKKOS_MEMPOOLLIST_INLINE
void MemPoolList::acquire_lock( size_t pos, Link * volatile * & freelist,
                                Link * & old_head ) const
{
  bool locked = false;

  while ( !locked ) {
    freelist = m_freelist + pos;
    old_head = *freelist;

    if ( old_head != KOKKOS_MEMPOOLLIST_LOCK ) {
      // In the initial look at the head, the freelist wasn't locked.
      // Attempt to lock the head of list.  If the list was changed (including
      // being locked) between the initial look and now, head will be different
      // than old_head.  This means the lock can't proceed and has to be
      // tried again.
      Link * const head = atomic_compare_exchange( freelist, old_head,
                                                   KOKKOS_MEMPOOLLIST_LOCK );

      if ( head == old_head ) locked = true;
    }
  }
}

KOKKOS_FUNCTION
KOKKOS_MEMPOOLLIST_INLINE
void MemPoolList::release_lock( Link * volatile * freelist, Link * const new_head ) const
{
#ifdef KOKKOS_MEMPOOLLIST_PRINTERR
  Link * const head =
#endif
    atomic_compare_exchange( freelist, KOKKOS_MEMPOOLLIST_LOCK, new_head );

#ifdef KOKKOS_MEMPOOLLIST_PRINTERR
  if ( head != KOKKOS_MEMPOOLLIST_LOCK ) {
    // We shouldn't get here, but this check is here for sanity.
    printf( "\n** MemoryPool::allocate() UNLOCK_ERROR(0x%lx) **\n",
            reinterpret_cast<unsigned long>( freelist ) );
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    fflush( stdout );
#endif
    Kokkos::abort( "" );
  }
#endif
}

KOKKOS_FUNCTION
KOKKOS_MEMPOOLLIST_INLINE
void * MemPoolList::allocate( size_t alloc_size ) const
{
  void * p = 0;

  // Find the first freelist whose chunk size is big enough for allocation.
  size_t l_exp = 0;
  while ( m_chunk_size[l_exp] > 0 && alloc_size > m_chunk_size[l_exp] ) ++l_exp;

#ifdef KOKKOS_MEMPOOLLIST_PRINTERR
  if ( m_chunk_size[l_exp] == 0 ) {
    Kokkos::abort( "\n** MemoryPool::allocate() REQUESTED_SIZE_TOO_LARGE **\n" );
  }
#endif

  // Do a fast fail test for an empty list.  This checks for l_exp and all
  // higher freelist pointers being 0.
  size_t l = l_exp;
  while ( m_chunk_size[l] > 0 && m_freelist[l] == 0 ) ++l;

  if ( m_chunk_size[l] != 0 ) {
    Link * volatile * l_exp_freelist = 0;
    Link * l_exp_old_head = 0;

    // Grab a lock on the l_exp freelist.
    acquire_lock( l_exp, l_exp_freelist, l_exp_old_head );

    if ( l_exp_old_head != 0 ) {
      // The l_exp freelist isn't empty.

      // Get a local copy of the second entry in the list.
      Link * const head_next =
        *reinterpret_cast<Link * volatile *>( &(l_exp_old_head->m_next) );

      // Release the lock, replacing the head with the next entry on the list.
      release_lock( l_exp_freelist, head_next );

      // Set the chunk to return.
      *reinterpret_cast<Link * volatile *>( &(l_exp_old_head->m_next) ) = 0;
      p = l_exp_old_head;
    }
    else {
      // The l_exp freelist is empty.

      l = l_exp + 1;
      bool done = false;

      while ( !done ) {
        // Find the next freelist that is either locked or not empty.  A locked
        // freelist will probably have memory available when the lock is
        // released.
        while ( m_chunk_size[l] > 0 && m_freelist[l] == 0 ) ++l;

        if ( m_chunk_size[l] == 0 ) {
          // We got to the end of the list of freelists without finding any
          // available memory which means the pool is empty.  Release the lock
          // on the l_exp freelist.  Put NULL as the head since the freelist
          // was empty.
          release_lock( l_exp_freelist, 0 );

          // Exit out of the loop.
          done = true;
        }
        else {
          Link * volatile * l_freelist = 0;
          Link * l_old_head = 0;

          // Grab a lock on the l freelist.
          acquire_lock( l, l_freelist, l_old_head );

          if ( l_old_head != 0 ) {
            // The l freelist has chunks.  Grab one to divide.

            // Subdivide the chunk into smaller chunks.  The first chunk will
            // be returned to satisfy the allocaiton request.  The remainder
            // of the chunks will be inserted onto the appropriate freelist.
            size_t num_chunks = m_chunk_size[l] / m_chunk_size[l_exp];
            char * pchar = reinterpret_cast<char *>( l_old_head );

            // Link the chunks following the first chunk to form a list.
            for ( size_t i = 2; i < num_chunks; ++i ) {
              Link * chunk =
                reinterpret_cast<Link *>( pchar + (i - 1) * m_chunk_size[l_exp] );

              chunk->m_next =
                reinterpret_cast<Link *>( pchar + i * m_chunk_size[l_exp] );
            }

            Link * lp_head =
              reinterpret_cast<Link *>( pchar + m_chunk_size[l_exp] );

            Link * lp_tail =
              reinterpret_cast<Link *>( pchar + (num_chunks - 1) * m_chunk_size[l_exp] );

            // Assign lp_tail->m_next to be NULL since the l_exp freelist was
            // empty.
            *reinterpret_cast<Link * volatile *>( &(lp_tail->m_next) ) = 0;

            memory_fence();

            // Get a local copy of the second entry in the list.
            Link * const head_next =
              *reinterpret_cast<Link * volatile *>( &(l_old_head->m_next) );

            // Release the lock on the l freelist.
            release_lock( l_freelist, head_next );

            // This thread already has the lock on the l_exp freelist, so just
            // release the lock placing the divided memory on the list.
            release_lock( l_exp_freelist, lp_head );

            // Set the chunk to return.
            *reinterpret_cast<Link * volatile *>( &(l_old_head->m_next) ) = 0;
            p = l_old_head;
            done = true;
          }
          else {
            // Release the lock on the l freelist.  Put NULL as the head since
            // the freelist was empty.
            release_lock( l_freelist, 0 );
          }
        }
      }
    }
  }

#ifdef KOKKOS_MEMPOOLLIST_PRINT_INFO
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
  long val = p == 0 ?
             *reinterpret_cast<volatile long *>( &m_count ) :
             Kokkos::atomic_fetch_add( &m_count, 1 );

  printf( "  allocate(): %6ld   size: %6lu    l: %2lu  %2lu   0x%lx\n", val,
          alloc_size, l_exp, l, reinterpret_cast<unsigned long>( p ) );
  fflush( stdout );
#else
  printf( "  allocate()   size: %6lu    l: %2lu  %2lu   0x%lx\n", alloc_size,
          l_exp, l, reinterpret_cast<unsigned long>( p ) );
#endif
#endif

#ifdef KOKKOS_MEMPOOLLIST_PRINTERR
  if ( p == 0 ) {
    printf( "** MemoryPool::allocate() NO_CHUNKS_BIG_ENOUGH **\n" );
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    fflush( stdout );
#endif
  }
#endif

  return p;
}

KOKKOS_FUNCTION
KOKKOS_MEMPOOLLIST_INLINE
void MemPoolList::deallocate( void * alloc_ptr, size_t alloc_size ) const
{
#ifdef KOKKOS_MEMPOOLLIST_PRINTERR
  // Verify that the pointer is controlled by this pool.
  {
    char * ap = static_cast<char *>( alloc_ptr );

    if ( ap < m_data || ap + alloc_size > m_data + m_data_size ) {
      printf( "\n** MemoryPool::deallocate() ADDRESS_OUT_OF_RANGE(0x%lx) **\n",
              reinterpret_cast<unsigned long>( alloc_ptr ) );
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
      fflush( stdout );
#endif
      Kokkos::abort( "" );
    }
  }
#endif

  // Determine which freelist to place deallocated memory on.
  size_t l = 0;
  while ( m_chunk_size[l] > 0 && alloc_size > m_chunk_size[l] ) ++l;

#ifdef KOKKOS_MEMPOOLLIST_PRINTERR
  if ( m_chunk_size[l] == 0 ) {
    printf( "\n** MemoryPool::deallocate() CHUNK_TOO_LARGE(%lu) **\n", alloc_size );
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    fflush( stdout );
#endif
    Kokkos::abort( "" );
  }
#endif

  Link * lp = static_cast<Link *>( alloc_ptr );

  // Insert a single chunk at the head of the freelist.
  Link * volatile * freelist = m_freelist + l;

  bool inserted = false;

  while ( !inserted ) {
    Link * const old_head = *freelist;

    if ( old_head != KOKKOS_MEMPOOLLIST_LOCK ) {
      // In the initial look at the head, the freelist wasn't locked.

      // Proactively assign lp->m_next assuming a successful insertion into
      // the list.
      *reinterpret_cast<Link * volatile *>( &(lp->m_next) ) = old_head;

      memory_fence();

      // Attempt to insert at head of list.  If the list was changed
      // (including being locked) between the initial look and now, head will
      // be different than old_head.  This means the insert can't proceed and
      // has to be tried again.
      Link * const head = atomic_compare_exchange( freelist, old_head, lp );

      if ( head == old_head ) inserted = true;
    }
  }

#ifdef KOKKOS_MEMPOOLLIST_PRINT_INFO
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
  long val = Kokkos::atomic_fetch_add( &m_count, -1 ) - 1;
  printf( "deallocate(): %6ld   size: %6lu    l: %2lu       0x%lx\n", val,
          alloc_size, l, reinterpret_cast<unsigned long>( alloc_ptr ) );
  fflush( stdout );
#else
  printf( "deallocate()   size: %6lu    l: %2lu       0x%lx\n", alloc_size, l,
          reinterpret_cast<unsigned long>( alloc_ptr ) );
#endif
#endif
}


} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

#undef KOKKOS_MEMPOOLLIST_LOCK
#undef KOKKOS_MEMPOOLLIST_INLINE

#ifdef KOKKOS_MEMPOOLLIST_PRINTERR
#undef KOKKOS_MEMPOOLLIST_PRINTERR
#endif

#ifdef KOKKOS_MEMPOOLLIST_PRINT_INFO
#undef KOKKOS_MEMPOOLLIST_PRINT_INFO
#endif

#endif /* #ifndef KOKKOS_MEMORYPOOL_CPP */

