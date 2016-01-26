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

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

template < class Device >
class MemoryPool ;

namespace Impl {

class MemPoolList {
private:

  typedef Impl::SharedAllocationTracker  Tracker ;

  template< class > friend class Kokkos::Experimental::MemoryPool ;

  Tracker            m_track ;
  void * volatile *  m_head_list ;
  size_t             m_chunk_size ;
  size_t             m_chunk_count ;

public:

 // Public for the 'ParallelFor' to call for initialization.

  ~MemPoolList() = default ;
  MemPoolList() = default ;
  MemPoolList( MemPoolList && ) = default ;
  MemPoolList( const MemPoolList & ) = default ;
  MemPoolList & operator = ( MemPoolList && ) = default ;
  MemPoolList & operator = ( const MemPoolList & ) = default ;

 KOKKOS_INLINE_FUNCTION
 void operator()( size_t i ) const
   {
     const size_t n = m_chunk_size / sizeof(void*) ;
     void * volatile * const lp = m_head_list + i * n ;
     // The last entry is null
     *lp = i <= m_chunk_count
         ? (void*) ( lp + n )
         : (void*) 0 ; 
   }

private:

  template< class MemorySpace , class ExecutionSpace>
  inline
  MemPoolList( const MemorySpace & arg_space 
             , const ExecutionSpace & 
             , size_t        arg_chunk
             , size_t        arg_total
             )
    : m_track()
    , m_head_list(0)
    , m_chunk_size(0)
    , m_chunk_count( arg_chunk * ( ( arg_total + arg_chunk - 1 ) / arg_chunk ) )
  {
    typedef Impl::SharedAllocationRecord< MemorySpace, void >  SharedRecord ;
    typedef Kokkos::RangePolicy< ExecutionSpace > Range ;

    // Force chunk size to be power of two,
    // mininum of 32 (mininum of 4 x 8byte pointers),
    // and at least arg_chunk size.

    for ( m_chunk_size = 32 ; m_chunk_size < arg_chunk ; m_chunk_size <<= 1 );

    // Force allocation to be a multiple of actual chunk_size.
    // Allocate an extra chunk for the head(s) of the free list(s).
    const size_t alloc_size = m_chunk_size * ( m_chunk_count + 1 );

    SharedRecord * rec =
      SharedRecord::allocate( arg_space, "mempool", alloc_size );

    m_track.assign_allocated_record_to_uninitialized( rec );

    m_head_list = reinterpret_cast< void ** >( rec->data() );

    // Initialize link list of free chunks

    Kokkos::Impl::ParallelFor< MemPoolList , Range >
      closure( *this , Range( 0 , m_chunk_count + 1 ) );

    closure.execute();

    ExecutionSpace::fence();
  }

  KOKKOS_FUNCTION
  void * allocate( size_t arg_size ) const ;

  KOKKOS_FUNCTION
  void deallocate( void * arg_alloc , size_t arg_size ) const ;
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
template < class Device >
class MemoryPool {
private:

  Impl::MemPoolList  m_freelist ;

  typedef typename Device::memory_space backend_memory_space;
  typedef typename Device::execution_space execution_space;
public:

  //! Tag this class as a kokkos memory space
  typedef MemoryPool  memory_space ;

  //------------------------------------

  MemoryPool() = default;
  MemoryPool( MemoryPool && rhs ) = default;
  MemoryPool( const MemoryPool & rhs ) = default;
  MemoryPool & operator = ( MemoryPool && ) = default;
  MemoryPool & operator = ( const MemoryPool & ) = default;
  ~MemoryPool() = default;

  /**\brief  Allocate memory pool */
  MemoryPool
    ( const backend_memory_space & arg_space /* From where to allocate the pool. */
    , size_t arg_chunk_size   /* Hand out memory in chunks of this size. */
    , size_t arg_total_size   /* Total size of the pool. */
    )
    : m_freelist( arg_space , execution_space(), arg_chunk_size , arg_total_size )
  {}

  unsigned chunk_size() const { return m_freelist.m_chunk_size ; }

  ///\brief  Claim chunks of untracked memory from the pool.
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void * allocate( const size_t alloc_size ) const
    { return m_freelist.allocate( alloc_size ); }

  ///\brief  Release claimed memory back into the pool
  /// Can only be called from device.
  KOKKOS_INLINE_FUNCTION
  void deallocate( void * const alloc_ptr,
                   const size_t alloc_size ) const
    { m_freelist.deallocate( alloc_ptr , alloc_size ); }
};

} // namespace Experimental
} // namespace Kokkos

#endif /* #define KOKKOS_MEMORYPOOL_HPP */

