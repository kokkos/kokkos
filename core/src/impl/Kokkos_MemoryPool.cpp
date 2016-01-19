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

#include <Kokkos_MemoryPool.hpp>
#include <Kokkos_Atomic.hpp>

#if ! defined( KOKKOS_MEMPOOLLIST_INLINE )

namespace Kokkos {
namespace Experimental {
namespace Impl {

#define MEMPOOL_LIST_LOCK  reinterpret_cast<void*>( ~uintptr_t(0) )

KOKKOS_FUNCTION
void * MemPoolList::allocate( size_t arg_size ) const
{
  // Requires requested size less than or equal chunk size
  if ( m_chunk_size < arg_size ) {
    Kokkos::abort("MemoryPool::allocate ARGUMENT ERROR");
  }

  void * p = 0 ;

  bool pending = true ;

  while ( pending ) {

    // If head is null then nothing left to allocate.
    pending = 0 != ( p = *m_head_list );

    if ( pending && ( p != MEMPOOL_LIST_LOCK ) ) {
      // In the initial look at the head, the freelist wasn't empty or
      // locked. Attempt to lock the head of list.  If the list was changed
      // (including being locked) between the initial look and now, head will
      // be different.  This means the removal can't proceed
      // and has to be tried again.

      pending = p != atomic_compare_exchange( m_head_list, p, MEMPOOL_LIST_LOCK );
    }
  }

  if ( p ) {
    // The lock succeeded.  Get a local copy of the next entry
    void * const head_next = *reinterpret_cast<void * volatile *>( p );

    // Replace the lock with the next entry on the list.
    if ( MEMPOOL_LIST_LOCK != atomic_compare_exchange( m_head_list, MEMPOOL_LIST_LOCK, head_next ) ) {
      Kokkos::abort("MemoryPool::allocate UNLOCK ERROR");
    }
  }

  return p ;
}

KOKKOS_FUNCTION
void MemPoolList::deallocate( void * arg_alloc , size_t arg_size ) const
{
  {
    // Requires original pointer, test that it is one of the chunks

    const size_t dist = ((char *)arg_alloc) - ((char*)m_head_list);

    if ( ( ( dist % m_chunk_size ) != 0 ) ||
         ( m_chunk_count < ( dist / m_chunk_size ) ) ||
         ( m_chunk_size < arg_size ) ) {
      Kokkos::abort("MemoryPool::deallocate ARGUMENT ERROR");
    }
  }

  bool pending = true ;

  while ( pending ) {

    void * const head = *m_head_list ;

    if ( head != MEMPOOL_LIST_LOCK ) {
      // In the initial look at the head, the freelist wasn't locked.

      // Proactively assign next pointer assuming a successful insertion into
      // the list.
      *reinterpret_cast<void*volatile*>(arg_alloc) = head ;

      memory_fence();

      // Attempt to insert at head of list.  If the list was changed
      // (including being locked) between the initial look and now, head will
      // be different than old_head.  This means the insert can't proceed and
      // has to be tried again.

      pending = head != atomic_compare_exchange( m_head_list, head, arg_alloc );
    }
  }
}

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

#endif


