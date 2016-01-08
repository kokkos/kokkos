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

//#include <impl/Kokkos_HostSpace.hpp>

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
template < class MemorySpace >
class MemoryPool {
public:
  /// \brief Structure used to create a linked list from the memory chunk.
  ///
  /// The allocated memory is cast to this type internally to create the lists.
  struct Link {
    Link * next;
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

  //! This memory space preferred device_type
  typedef typename MemorySpace::device_type device_type;

  typedef typename MemorySpace::size_type                    size_type;
  typedef Impl::SharedAllocationRecord< MemorySpace, void >  SharedRecord;
  typedef Impl::SharedAllocationTracker                      Tracker;
  typedef Kokkos::View< Link, execution_space >              LinkView;

  //------------------------------------

//  MemoryPool() : m_freelist( "freelist") {}
  MemoryPool() = default;
  MemoryPool( MemoryPool && rhs ) = default;
  MemoryPool( const MemoryPool & rhs ) = default;
  MemoryPool & operator = ( MemoryPool && ) = default;
  MemoryPool & operator = ( const MemoryPool & ) = default;
  ~MemoryPool() = default;

  /**\brief  Default memory space instance */
  MemoryPool( const MemorySpace & memspace, /* From where to allocate the pool */
              size_type chunk_size, /* Hand out memory in chunks of this size */
              size_type total_size /* Total size of the pool */
            )
    : m_freelist( "freelist" )
  {
    // TODO: Ask Carter about how to better handle this.
    if ( chunk_size < 8 ) {
      std::cerr << "Chunk size must be at least 8 bytes.  Setting to be "
                << "8 bytes." << std::endl;
      chunk_size = 8;
    }

    // Force total_size to be a multiple of chunk_size.
    total_size = ( ( total_size + chunk_size - 1 ) / chunk_size ) * chunk_size;

    size_type num_chunks = total_size / chunk_size;

    SharedRecord * rec = SharedRecord::allocate( memspace, "mempool", total_size );
    m_pool_tracker.assign_allocated_record_to_uninitialized( rec );

    Link alink;
    alink.next = static_cast< Link * >( rec->data() );
    deep_copy(m_freelist, alink);

    char * head = static_cast< char * >( rec->data() );

    // TODO: The next two loops need to be done in parallel so they access the
    //       correct memory space.

    // Initialize all next pointers to 0.  This is a bit overkill since we only
    // need to set the next pointer of the last chunk to 0, but two simple loops
    // should be faster than one loop with an if statement.
    for ( size_type i = 0; i < num_chunks; ++i )
    {
      Link * lp = reinterpret_cast< Link * >( head + i * chunk_size );
      lp->next = 0;
    }

    // Initialize the next pointers to point to the next chunk for all but the
    // last chunk.
    for ( size_type i = 1; i < num_chunks; ++i )
    {
      Link * lp = reinterpret_cast< Link * >( head + ( i - 1 ) * chunk_size );
      lp->next = reinterpret_cast< Link * >( head + i * chunk_size );
    }

//    printf(" Pool size: %llu\n", rec->size());
  }

  ///\brief  Claim chunks of untracked memory from the pool.
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void * allocate( const size_t alloc_size ) const
  {
    void * p = static_cast< void * >((*m_freelist).next);
    (*m_freelist).next = (*m_freelist).next->next;
    return p;
  }

  ///\brief  Release claimed memory back into the pool
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void deallocate( void * const alloc_ptr,
                   const size_t alloc_size ) const
  {
    Link* lp = static_cast< Link * >(alloc_ptr);
    lp->next = (*m_freelist).next;
    (*m_freelist).next = lp;
  }

private:
  Tracker      m_pool_tracker;
  LinkView     m_freelist;
};

} // namespace Experimental
} // namespace Kokkos

#endif /* #define KOKKOS_MEMORYPOOL_HPP */
