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
#include <Kokkos_MemoryTraits.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>

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

  //! Tag this class as a kokkos memory space
  typedef MemoryPool                       memory_space ;
  typedef typename MemorySpace::size_type  size_type ;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
  typedef typename MemorySpace::execution_space execution_space ;

  //! This memory space preferred device_type
  typedef typename MemorySpace::device_type device_type;

  //------------------------------------

  MemoryPool() = default ;
  MemoryPool( MemoryPool && rhs ) = default ;
  MemoryPool( const MemoryPool & rhs ) = default ;
  MemoryPool & operator = ( MemoryPool && ) = default ;
  MemoryPool & operator = ( const MemoryPool & ) = default ;
  ~MemoryPool() = default ;

  /**\brief  Default memory space instance */
  MemoryPool( const MemorySpace & space /* From where to allocate the pool */
            , size_type chunk_size /* Hand out memory in chunks of this size */
            , size_type total_size /* Total size of the pool */
            )
    : m_memory_space( space )
    , m_chunk_mask( 0 )
    {
      size_type i = 1 ;
      while ( total_size < i ) i <<= 1 ;
      m_chunk_mask = i - 1 ;
    }

  /**\brief  Claim chunks of untracked memory from the pool. */
  void * allocate( const size_t alloc_size ) const
    {
      // Improper implementation
      return m_memory_space.allocate( ( alloc_size + m_chunk_mask ) & ~m_chunk_mask );
    }

  /**\brief  Release claimed memory back into the pool */
  void deallocate( void * const alloc_ptr ,
                   const size_t alloc_size ) const
    { m_memory_space.deallocate( alloc_ptr , alloc_size ); }

private:
  MemorySpace  m_memory_space ;
  size_type    m_chunk_mask ;
};

} // namespace Experimental
} // namespace Kokkos

#endif /* #define KOKKOS_MEMORYPOOL_HPP */
