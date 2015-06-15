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

#include <memory.h>
#include <stddef.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <cstring>

#include <Kokkos_HostSpace.hpp>
#include <impl/Kokkos_BasicAllocators.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Kokkos_Atomic.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {


DeepCopy<HostSpace,HostSpace>::DeepCopy( void * dst , const void * src , size_t n )
{
  memcpy( dst , src , n );
}

}
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace {

static const int QUERY_SPACE_IN_PARALLEL_MAX = 16 ;

typedef int (* QuerySpaceInParallelPtr )();

QuerySpaceInParallelPtr s_in_parallel_query[ QUERY_SPACE_IN_PARALLEL_MAX ] ;
int s_in_parallel_query_count = 0 ;

} // namespace <empty>

void HostSpace::register_in_parallel( int (*device_in_parallel)() )
{
  if ( 0 == device_in_parallel ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::HostSpace::register_in_parallel ERROR : given NULL" ) );
  }

  int i = -1 ;

  if ( ! (device_in_parallel)() ) {
    for ( i = 0 ; i < s_in_parallel_query_count && ! (*(s_in_parallel_query[i]))() ; ++i );
  }

  if ( i < s_in_parallel_query_count ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::HostSpace::register_in_parallel_query ERROR : called in_parallel" ) );

  }

  if ( QUERY_SPACE_IN_PARALLEL_MAX <= i ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::HostSpace::register_in_parallel_query ERROR : exceeded maximum" ) );

  }

  for ( i = 0 ; i < s_in_parallel_query_count && s_in_parallel_query[i] != device_in_parallel ; ++i );

  if ( i == s_in_parallel_query_count ) {
    s_in_parallel_query[s_in_parallel_query_count++] = device_in_parallel ;
  }
}

int HostSpace::in_parallel()
{
  const int n = s_in_parallel_query_count ;

  int i = 0 ;

  while ( i < n && ! (*(s_in_parallel_query[i]))() ) { ++i ; }

  return i < n ;
}

} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

Impl::AllocationTracker HostSpace::allocate_and_track( const std::string & label, const size_t size )
{
  return Impl::AllocationTracker( allocator(), size, label );
}

} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {

HostSpace::HostSpace()
{
}

void * HostSpace::allocate( const size_t arg_alloc_size ) const
{
  constexpr size_t alignment = 128 ;

  void * ptr = NULL;
  if ( arg_alloc_size ) {
#if defined( __INTEL_COMPILER ) && !defined ( KOKKOS_HAVE_CUDA )
    ptr = _mm_malloc( arg_alloc_size , alignment );

#elif ( defined( _POSIX_C_SOURCE ) && _POSIX_C_SOURCE >= 200112L ) || \
    ( defined( _XOPEN_SOURCE )   && _XOPEN_SOURCE   >= 600 )

    posix_memalign( & ptr, alignment , arg_alloc_size );

#else
    // Over-allocate to and round up to guarantee proper alignment.
    size_t size_padded = arg_alloc_size + alignment + sizeof(void *);
    void * alloc_ptr = malloc( size_padded );

    if (alloc_ptr) {
      uintptr_t address = reinterpret_cast<uintptr_t>(alloc_ptr);
      // offset enough to record the alloc_ptr
      address += sizeof(void *);
      uintptr_t rem = address % alignment;
      uintptr_t offset = rem ? (alignment - rem) : 0u;
      address += offset;
      ptr = reinterpret_cast<void *>(address);
      // record the alloc'd pointer
      address -= sizeof(void *);
      *reinterpret_cast<void **>(address) = alloc_ptr;
    }
#endif
  }
  return ptr;
}

void HostSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
  if ( arg_alloc_ptr ) {
#if defined( __INTEL_COMPILER ) && !defined ( KOKKOS_HAVE_CUDA )
    _mm_free( arg_alloc_ptr );

#elif ( defined( _POSIX_C_SOURCE ) && _POSIX_C_SOURCE >= 200112L ) || \
      ( defined( _XOPEN_SOURCE )   && _XOPEN_SOURCE   >= 600 )
    free( arg_alloc_ptr );
#else
    // get the alloc'd pointer
    void * alloc_ptr = *(reinterpret_cast<void **>(ptr) -1);
    free( alloc_ptr );
#endif
  }
}

} // namespace Kokkos

namespace Kokkos {
namespace Experimental {
namespace Impl {

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::HostSpace , void >::s_root_record ;

void
SharedAllocationRecord< Kokkos::HostSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::HostSpace , void >::
~SharedAllocationRecord()
{
  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::HostSpace , void >::
SharedAllocationRecord( const Kokkos::HostSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      ( & SharedAllocationRecord< Kokkos::HostSpace , void >::s_root_record
      , reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
}

SharedAllocationRecord< Kokkos::HostSpace , void > *
SharedAllocationRecord< Kokkos::HostSpace , void >::get_record( void * alloc_ptr )
{
  using Header     = SharedAllocationHeader ;
  using RecordBase = SharedAllocationRecord< void , void > ;
  using RecordHost = SharedAllocationRecord< Kokkos::HostSpace , void > ;

  SharedAllocationHeader const * const head   = Header::get_header( alloc_ptr );
  RecordHost                   * const record = static_cast< RecordHost * >( head->m_record );

  if ( record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::HostSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< Kokkos::HostSpace , void >::
print_records( std::ostream & s , const Kokkos::HostSpace & space , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "HostSpace" , & s_root_record , detail );
}

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace {
  const unsigned HOST_SPACE_ATOMIC_MASK = 0xFFFF;
  const unsigned HOST_SPACE_ATOMIC_XOR_MASK = 0x5A39;
  static int HOST_SPACE_ATOMIC_LOCKS[HOST_SPACE_ATOMIC_MASK+1];
}

namespace Impl {
void init_lock_array_host_space() {
  static int is_initialized = 0;
  if(! is_initialized)
    for(int i = 0; i < static_cast<int> (HOST_SPACE_ATOMIC_MASK+1); i++)
      HOST_SPACE_ATOMIC_LOCKS[i] = 0;
}

bool lock_address_host_space(void* ptr) {
  return 0 == atomic_compare_exchange( &HOST_SPACE_ATOMIC_LOCKS[
      (( size_t(ptr) >> 2 ) & HOST_SPACE_ATOMIC_MASK) ^ HOST_SPACE_ATOMIC_XOR_MASK] ,
                                  0 , 1);
}

void unlock_address_host_space(void* ptr) {
   atomic_exchange( &HOST_SPACE_ATOMIC_LOCKS[
      (( size_t(ptr) >> 2 ) & HOST_SPACE_ATOMIC_MASK) ^ HOST_SPACE_ATOMIC_XOR_MASK] ,
                    0);
}

}
}
