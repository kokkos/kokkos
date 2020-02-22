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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <algorithm>
#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include <Kokkos_SICMSpace.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Kokkos_Atomic.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

static void arena_deleter(sicm_arena * arena)
{
    sicm_arena_destroy(*arena);
    delete arena;
}

/* Default allocation mechanism */
SICMSpace::SICMSpace()
  : arena( new sicm_arena(ARENA_DEFAULT), arena_deleter )
{}

SICMSpace::SICMSpace( sicm_device_list * devices )
  : SICMSpace()
{
  if ( devices ) {
    arena = std::shared_ptr <sicm_arena> (
      new sicm_arena(sicm_arena_create(0, static_cast<sicm_arena_flags>(0), devices)),
      arena_deleter
    );
  }

  if ( *arena == ARENA_DEFAULT ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::SICMSpace Failed to create arena") );
  }
}

void * SICMSpace::allocate( const size_t arg_alloc_size ) const
{
  static_assert( sizeof(void*) == sizeof(uintptr_t)
               , "Error sizeof(void*) != sizeof(uintptr_t)" );

  void * ptr = nullptr;
  if ( *arena == ARENA_DEFAULT ) {
    ptr = sicm_alloc(arg_alloc_size);
  }
  else {
    ptr = sicm_arena_alloc(*arena, arg_alloc_size);
  }

  if ( ptr == nullptr ) {
    std::ostringstream msg ;
    msg << "Kokkos::SICMSpace::allocate"
        << "( " << arg_alloc_size << " ) FAILED: NULL" ;

    std::cerr << msg.str() << std::endl ;

    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  return ptr;
}

void SICMSpace::deallocate( void * const arg_alloc_ptr
    , const size_t
    ) const
{
  sicm_free(arg_alloc_ptr);
}

} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::s_root_record ;
#endif

void
SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::Experimental::SICMSpace::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif

  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
SharedAllocationRecord( const Kokkos::Experimental::SICMSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      (
#ifdef KOKKOS_DEBUG
      & SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::s_root_record,
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
   }
#endif
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char) 0;
}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
allocate_tracked( const Kokkos::Experimental::SICMSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<Experimental::SICMSpace,Experimental::SICMSpace>( r_new->data() , r_old->data()
                                                                           , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void > *
SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >  RecordSICM ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordSICM                   * const record = head ? static_cast< RecordSICM * >( head->m_record ) : (RecordSICM *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
#ifdef KOKKOS_DEBUG
void SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
print_records( std::ostream & s , const Kokkos::Experimental::SICMSpace & , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "SICMSpace" , & s_root_record , detail );
}
#else
void SharedAllocationRecord< Kokkos::Experimental::SICMSpace , void >::
print_records( std::ostream & , const Kokkos::Experimental::SICMSpace & , bool )
{
  throw_runtime_exception("SharedAllocationRecord<SICMSpace>::print_records only works with KOKKOS_DEBUG enabled");
}
#endif

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace {
  const unsigned SICM_SPACE_ATOMIC_MASK = 0xFFFF;
  const unsigned SICM_SPACE_ATOMIC_XOR_MASK = 0x5A39;
  static int SICM_SPACE_ATOMIC_LOCKS[SICM_SPACE_ATOMIC_MASK+1];
}

namespace Impl {
void init_lock_array_sicm_space() {
  static int is_initialized = 0;
  if(! is_initialized)
    for(int i = 0; i < static_cast<int> (SICM_SPACE_ATOMIC_MASK+1); i++)
      SICM_SPACE_ATOMIC_LOCKS[i] = 0;
}

bool lock_address_sicm_space(void* ptr) {
#if defined( KOKKOS_ENABLE_ISA_X86_64 ) && defined ( KOKKOS_ENABLE_TM )
  const unsigned status = _xbegin();

  if( _XBEGIN_STARTED == status ) {
	const int val = SICM_SPACE_ATOMIC_LOCKS[(( size_t(ptr) >> 2 ) &
		SICM_SPACE_ATOMIC_MASK) ^ SICM_SPACE_ATOMIC_XOR_MASK];

	if( 0 == val ) {
		SICM_SPACE_ATOMIC_LOCKS[(( size_t(ptr) >> 2 ) &
                   SICM_SPACE_ATOMIC_MASK) ^ SICM_SPACE_ATOMIC_XOR_MASK] = 1;
	} else {
		_xabort( 1 );
	}

	_xend();

	return 1;
  } else {
#endif
  return 0 == atomic_compare_exchange( &SICM_SPACE_ATOMIC_LOCKS[
      (( size_t(ptr) >> 2 ) & SICM_SPACE_ATOMIC_MASK) ^ SICM_SPACE_ATOMIC_XOR_MASK] ,
                                  0 , 1);
#if defined( KOKKOS_ENABLE_ISA_X86_64 ) && defined ( KOKKOS_ENABLE_TM )
  }
#endif
}

void unlock_address_sicm_space(void* ptr) {
#if defined( KOKKOS_ENABLE_ISA_X86_64 ) && defined ( KOKKOS_ENABLE_TM )
  const unsigned status = _xbegin();

  if( _XBEGIN_STARTED == status ) {
	SICM_SPACE_ATOMIC_LOCKS[(( size_t(ptr) >> 2 ) &
        	SICM_SPACE_ATOMIC_MASK) ^ SICM_SPACE_ATOMIC_XOR_MASK] = 0;
  } else {
#endif
   atomic_exchange( &SICM_SPACE_ATOMIC_LOCKS[
      (( size_t(ptr) >> 2 ) & SICM_SPACE_ATOMIC_MASK) ^ SICM_SPACE_ATOMIC_XOR_MASK] ,
                    0);
#if defined( KOKKOS_ENABLE_ISA_X86_64 ) && defined ( KOKKOS_ENABLE_TM )
  }
#endif
}

}
}
