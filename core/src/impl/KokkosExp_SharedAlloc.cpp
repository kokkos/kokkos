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

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

int SharedAllocationRecord< void , void >::s_tracking_enabled = 1 ;

void SharedAllocationRecord< void , void >::tracking_claim_and_disable()
{
  // A host thread claim and disable tracking flag

  while ( ! Kokkos::atomic_compare_exchange_strong( & s_tracking_enabled, 1, 0 ) );
}

void SharedAllocationRecord< void , void >::tracking_release_and_enable()
{
  // The host thread that claimed and disabled the tracking flag
  // now release and enable tracking.

  if ( ! Kokkos::atomic_compare_exchange_strong( & s_tracking_enabled, 0, 1 ) ){
    Kokkos::Impl::throw_runtime_exception("Kokkos::Experimental::Impl::SharedAllocationRecord<>::tracking_release_and_enable FAILED, this host process thread did not hold the lock" );
  }
}

//----------------------------------------------------------------------------

bool
SharedAllocationRecord< void , void >::
is_sane( SharedAllocationRecord< void , void > * arg_record )
{
  constexpr static SharedAllocationRecord * zero = 0 ;

  SharedAllocationRecord * const root = arg_record ? arg_record->m_root : 0 ;

  bool ok = root != 0 && root->m_count == 0 ;

  if ( ok ) {
    SharedAllocationRecord * root_next = 0 ;

    // Lock the list:
    while ( ( root_next = Kokkos::atomic_exchange( & root->m_next , zero ) ) == zero );

    for ( SharedAllocationRecord * rec = root_next ; ok && rec != root ; rec = rec->m_next ) {
      const bool ok_non_null  = rec && rec->m_prev && ( rec == root || rec->m_next );
      const bool ok_root      = ok_non_null && rec->m_root == root ;
      const bool ok_prev_next = ok_non_null && ( rec->m_prev != root ? rec->m_prev->m_next == rec : root_next == rec );
      const bool ok_next_prev = ok_non_null && rec->m_next->m_prev == rec ;
      const bool ok_count     = ok_non_null && 0 <= rec->m_count ;

      ok = ok_root && ok_prev_next && ok_next_prev && ok_count ;

if ( ! ok ) {
  //Formatting dependent on sizeof(uintptr_t) 
  const char * format_string;
  
  if (sizeof(uintptr_t) == sizeof(unsigned long)) {
     format_string = "Kokkos::Experimental::Impl::SharedAllocationRecord failed is_sane: rec(0x%.12lx){ m_count(%d) m_root(0x%.12lx) m_next(0x%.12lx) m_prev(0x%.12lx) m_next->m_prev(0x%.12lx) m_prev->m_next(0x%.12lx) }\n";
  }
  else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
     format_string = "Kokkos::Experimental::Impl::SharedAllocationRecord failed is_sane: rec(0x%.12llx){ m_count(%d) m_root(0x%.12llx) m_next(0x%.12llx) m_prev(0x%.12llx) m_next->m_prev(0x%.12llx) m_prev->m_next(0x%.12llx) }\n";
  }

  fprintf(stderr
        , format_string 
        , reinterpret_cast< uintptr_t >( rec )
        , rec->m_count
        , reinterpret_cast< uintptr_t >( rec->m_root )
        , reinterpret_cast< uintptr_t >( rec->m_next )
        , reinterpret_cast< uintptr_t >( rec->m_prev )
        , reinterpret_cast< uintptr_t >( rec->m_next->m_prev )
        , reinterpret_cast< uintptr_t >( rec->m_prev != rec->m_root ? rec->m_prev->m_next : root_next )
        );
}

    }

    if ( zero != Kokkos::atomic_exchange( & root->m_next , root_next ) ) {
      Kokkos::Impl::throw_runtime_exception("Kokkos::Experimental::Impl::SharedAllocationRecord failed is_sane unlocking");
    }
  }

  return ok ; 
}

SharedAllocationRecord<void,void> *
SharedAllocationRecord<void,void>::find( SharedAllocationRecord<void,void> * const arg_root , void * const arg_data_ptr )
{
  constexpr static SharedAllocationRecord * zero = 0 ;

  SharedAllocationRecord * root_next = 0 ;

  // Lock the list:
  while ( ( root_next = Kokkos::atomic_exchange( & arg_root->m_next , zero ) ) == zero );

  // Iterate searching for the record with this data pointer

  SharedAllocationRecord * r = root_next ;

  while ( ( r != arg_root ) && ( r->data() != arg_data_ptr ) ) { r = r->m_next ; }

  if ( r == arg_root ) { r = 0 ; }

  if ( zero != Kokkos::atomic_exchange( & arg_root->m_next , root_next ) ) {
    Kokkos::Impl::throw_runtime_exception("Kokkos::Experimental::Impl::SharedAllocationRecord failed locking/unlocking");
  }

  return r ;
}


/**\brief  Construct and insert into 'arg_root' tracking set.
 *         use_count is zero.
 */
SharedAllocationRecord< void , void >::
SharedAllocationRecord( SharedAllocationRecord<void,void> * arg_root
                      , SharedAllocationHeader            * arg_alloc_ptr
                      , size_t                              arg_alloc_size
                      , SharedAllocationRecord< void , void >::function_type  arg_dealloc
                      )
  : m_alloc_ptr(  arg_alloc_ptr )
  , m_alloc_size( arg_alloc_size )
  , m_dealloc(    arg_dealloc )
  , m_root( arg_root )
  , m_prev( 0 )
  , m_next( 0 )
  , m_count( 0 )
{
  constexpr static SharedAllocationRecord * zero = 0 ;

  // Insert into the root double-linked list for tracking
  //
  // before:  arg_root->m_next == next ; next->m_prev == arg_root
  // after:   arg_root->m_next == this ; this->m_prev == arg_root ;
  //              this->m_next == next ; next->m_prev == this

  m_prev = m_root ;

  // Read root->m_next and lock by setting to zero
  while ( ( m_next = Kokkos::atomic_exchange( & m_root->m_next , zero ) ) == zero );

  m_next->m_prev = this ;

  if ( zero != Kokkos::atomic_exchange( & m_root->m_next , this ) ) {
    Kokkos::Impl::throw_runtime_exception("Kokkos::Experimental::Impl::SharedAllocationRecord failed locking/unlocking");
  }
}

void
SharedAllocationRecord< void , void >::
increment( SharedAllocationRecord< void , void > * arg_record )
{
  const int old_count = Kokkos::atomic_fetch_add( & arg_record->m_count , 1 );

  if ( old_count < 0 ) { // Error
    Kokkos::Impl::throw_runtime_exception("Kokkos::Experimental::Impl::SharedAllocationRecord failed increment");
  }
}

SharedAllocationRecord< void , void > *
SharedAllocationRecord< void , void >::
decrement( SharedAllocationRecord< void , void > * arg_record )
{
  constexpr static SharedAllocationRecord * zero = 0 ;

  const int old_count = Kokkos::atomic_fetch_add( & arg_record->m_count , -1 );

  if ( old_count == 1 ) {

    // before:  arg_record->m_prev->m_next == arg_record  &&
    //          arg_record->m_next->m_prev == arg_record
    //
    // after:   arg_record->m_prev->m_next == arg_record->m_next  &&
    //          arg_record->m_next->m_prev == arg_record->m_prev

    SharedAllocationRecord * root_next = 0 ;

    // Lock the list:
    while ( ( root_next = Kokkos::atomic_exchange( & arg_record->m_root->m_next , zero ) ) == zero );

    arg_record->m_next->m_prev = arg_record->m_prev ;

    if ( root_next != arg_record ) {
      arg_record->m_prev->m_next = arg_record->m_next ;
    }
    else {
      // before:  arg_record->m_root == arg_record->m_prev
      // after:   arg_record->m_root == arg_record->m_next
      root_next = arg_record->m_next ; 
    }

    // Unlock the list:
    if ( zero != Kokkos::atomic_exchange( & arg_record->m_root->m_next , root_next ) ) {
      Kokkos::Impl::throw_runtime_exception("Kokkos::Experimental::Impl::SharedAllocationRecord failed decrement unlocking");
    }

    arg_record->m_next = 0 ;
    arg_record->m_prev = 0 ;

    function_type d = arg_record->m_dealloc ;
    (*d)( arg_record );
    arg_record = 0 ;
  }
  else if ( old_count < 1 ) { // Error
    Kokkos::Impl::throw_runtime_exception("Kokkos::Experimental::Impl::SharedAllocationRecord failed decrement count");
  }

  return arg_record ;
}

void
SharedAllocationRecord< void , void >::
print_host_accessible_records( std::ostream & s
                             , const char * const space_name
                             , const SharedAllocationRecord * const root
                             , const bool detail )
{
  const SharedAllocationRecord< void , void > * r = root ;

  char buffer[256] ;

  if ( detail ) {
    do {
      //Formatting dependent on sizeof(uintptr_t) 
      const char * format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string = "%s addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx + %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      }
      else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string = "%s addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ 0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf( buffer , 256
              , format_string
              , space_name
              , reinterpret_cast<uintptr_t>( r )
              , reinterpret_cast<uintptr_t>( r->m_prev )
              , reinterpret_cast<uintptr_t>( r->m_next )
              , reinterpret_cast<uintptr_t>( r->m_alloc_ptr )
              , r->m_alloc_size
              , r->m_count
              , reinterpret_cast<uintptr_t>( r->m_dealloc )
              , r->m_alloc_ptr->m_label
              );
      std::cout << buffer ;
      r = r->m_next ;
    } while ( r != root );
  }
  else {
    do {
      if ( r->m_alloc_ptr ) {
        //Formatting dependent on sizeof(uintptr_t) 
        const char * format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) { 
          format_string = "%s [ 0x%.12lx + %ld ] %s\n";
        }
        else if (sizeof(uintptr_t) == sizeof(unsigned long long)) { 
          format_string = "%s [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf( buffer , 256
                , format_string
                , space_name
                , reinterpret_cast< uintptr_t >( r->data() )
                , r->size()
                , r->m_alloc_ptr->m_label
                );
      }
      else {
        snprintf( buffer , 256 , "%s [ 0 + 0 ]\n" , space_name );
      }
      std::cout << buffer ;
      r = r->m_next ;
    } while ( r != root );
  }
}

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */


