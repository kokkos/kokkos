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

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <atomic>

#include <Kokkos_Core.hpp>
#include <Kokkos_Cuda.hpp>
#include <Kokkos_ResCudaSpace.hpp>

#include <Cuda/Kokkos_Cuda_Instance.hpp>
#include <impl/Kokkos_Error.hpp>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif


/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

DeepCopy<ResCudaSpace,ResCudaSpace,Cuda>::DeepCopy( void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpy( dst , src , n , cudaMemcpyDefault ) ); }

DeepCopy<HostSpace,ResCudaSpace,Cuda>::DeepCopy( void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpy( dst , src , n , cudaMemcpyDefault ) ); }

DeepCopy<ResCudaSpace,HostSpace,Cuda>::DeepCopy( void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpy( dst , src , n , cudaMemcpyDefault ) ); }

DeepCopy<ResCudaSpace,ResCudaSpace,Cuda>::DeepCopy( const Cuda & instance , void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpyAsync( dst , src , n , cudaMemcpyDefault , instance.cuda_stream() ) ); }

DeepCopy<HostSpace,ResCudaSpace,Cuda>::DeepCopy( const Cuda & instance , void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpyAsync( dst , src , n , cudaMemcpyDefault , instance.cuda_stream() ) ); }

DeepCopy<ResCudaSpace,HostSpace,Cuda>::DeepCopy( const Cuda & instance , void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpyAsync( dst , src , n , cudaMemcpyDefault , instance.cuda_stream() ) ); }

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/



/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {


ResCudaSpace::ResCudaSpace()
  : CudaSpace()
{
}

std::map<std::string, Kokkos::Experimental::DuplicateTracker * > ResCudaSpace::duplicate_map;

void ResCudaSpace::clear_duplicates_list() {
   std::map<std::string, Kokkos::Experimental::DuplicateTracker* >::iterator it = ResCudaSpace::duplicate_map.begin();
   while ( it != ResCudaSpace::duplicate_map.end() ) {
       Kokkos::Experimental::DuplicateTracker* dt = static_cast<Kokkos::Experimental::DuplicateTracker*>(it->second);
       delete dt;
       duplicate_map.erase(it);
   }
}
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::ResCudaSpace , void >::s_root_record ;

#endif

::cudaTextureObject_t
SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
attach_texture_object( const unsigned sizeof_alias
                     , void *   const alloc_ptr
                     , size_t   const alloc_size )
{
  enum { TEXTURE_BOUND_1D = 1u << 27 };

  if ( ( alloc_ptr == 0 ) || ( sizeof_alias * TEXTURE_BOUND_1D <= alloc_size ) ) {
    std::ostringstream msg ;
    msg << "Kokkos::ResCudaSpace ERROR: Cannot attach texture object to"
        << " alloc_ptr(" << alloc_ptr << ")"
        << " alloc_size(" << alloc_size << ")"
        << " max_size(" << ( sizeof_alias * TEXTURE_BOUND_1D ) << ")" ;
    std::cerr << msg.str() << std::endl ;
    std::cerr.flush();
    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  ::cudaTextureObject_t tex_obj ;

  struct cudaResourceDesc resDesc ;
  struct cudaTextureDesc  texDesc ;

  memset( & resDesc , 0 , sizeof(resDesc) );
  memset( & texDesc , 0 , sizeof(texDesc) );

  resDesc.resType                = cudaResourceTypeLinear ;
  resDesc.res.linear.desc        = ( sizeof_alias ==  4 ?  cudaCreateChannelDesc< int >() :
                                   ( sizeof_alias ==  8 ?  cudaCreateChannelDesc< ::int2 >() :
                                  /* sizeof_alias == 16 */ cudaCreateChannelDesc< ::int4 >() ) );
  resDesc.res.linear.sizeInBytes = alloc_size ;
  resDesc.res.linear.devPtr      = alloc_ptr ;

  CUDA_SAFE_CALL( cudaCreateTextureObject( & tex_obj , & resDesc, & texDesc, NULL ) );

  return tex_obj ;
}

std::string
SharedAllocationRecord< Kokkos::ResCudaSpace , void >::get_label() const
{
  SharedAllocationHeader header ;

  Kokkos::Impl::DeepCopy< Kokkos::HostSpace , Kokkos::ResCudaSpace >( & header , RecordBase::head() , sizeof(SharedAllocationHeader) );

  return std::string( header.m_label );
}

SharedAllocationRecord< Kokkos::ResCudaSpace , void > *
SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
allocate( const Kokkos::ResCudaSpace &  arg_space
        , const std::string       &  arg_label
        , const size_t               arg_alloc_size
        )
{
  return new SharedAllocationRecord( arg_space , arg_label , arg_alloc_size );
}

void
SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {

    SharedAllocationHeader header ;
    Kokkos::Impl::DeepCopy<ResCudaSpace,HostSpace>( & header , RecordBase::m_alloc_ptr , sizeof(SharedAllocationHeader) );

    Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(Kokkos::ResCudaSpace::name()),header.m_label,
      data(),size());
  }
  #endif

  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
SharedAllocationRecord( const Kokkos::ResCudaSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      (
#ifdef KOKKOS_DEBUG
        & SharedAllocationRecord< Kokkos::ResCudaSpace , void >::s_root_record,
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_tex_obj( 0 )
  , m_space( arg_space )
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
  }
  #endif

  SharedAllocationHeader header ;

  // Fill in the Header information
  header.m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( header.m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char) 0;

  // Copy to device memory
  Kokkos::Impl::DeepCopy<ResCudaSpace,HostSpace>( RecordBase::m_alloc_ptr , & header , sizeof(SharedAllocationHeader) );
}


//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
allocate_tracked( const Kokkos::ResCudaSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<ResCudaSpace,ResCudaSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

//----------------------------------------------------------------------------

SharedAllocationRecord< Kokkos::ResCudaSpace , void > *
SharedAllocationRecord< Kokkos::ResCudaSpace , void >::get_record( void * alloc_ptr )
{
  using RecordCuda = SharedAllocationRecord< Kokkos::ResCudaSpace , void > ;

  using Header     = SharedAllocationHeader ;

  // Copy the header from the allocation
  Header head ;

  Header const * const head_cuda = alloc_ptr ? Header::get_header( alloc_ptr ) : (Header*) 0 ;

  if ( alloc_ptr ) {
    Kokkos::Impl::DeepCopy<HostSpace,ResCudaSpace>( & head , head_cuda , sizeof(SharedAllocationHeader) );
  }

  RecordCuda * const record = alloc_ptr ? static_cast< RecordCuda * >( head.m_record ) : (RecordCuda *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head_cuda ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::ResCudaSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void
SharedAllocationRecord< Kokkos::ResCudaSpace , void >::
print_records( std::ostream & s , const Kokkos::ResCudaSpace & , bool detail )
{
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord< void , void > * r = & s_root_record ;

  char buffer[256] ;

  SharedAllocationHeader head ;

  if ( detail ) {
    do {
      if ( r->m_alloc_ptr ) {
        Kokkos::Impl::DeepCopy<HostSpace,ResCudaSpace>( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );
      }
      else {
        head.m_label[0] = 0 ;
      }

      //Formatting dependent on sizeof(uintptr_t)
      const char * format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string = "Cuda addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx + %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      }
      else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string = "Cuda addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ 0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf( buffer , 256
              , format_string
              , reinterpret_cast<uintptr_t>( r )
              , reinterpret_cast<uintptr_t>( r->m_prev )
              , reinterpret_cast<uintptr_t>( r->m_next )
              , reinterpret_cast<uintptr_t>( r->m_alloc_ptr )
              , r->m_alloc_size
              , r->m_count
              , reinterpret_cast<uintptr_t>( r->m_dealloc )
              , head.m_label
              );
      s << buffer ;
      r = r->m_next ;
    } while ( r != & s_root_record );
  }
  else {
    do {
      if ( r->m_alloc_ptr ) {

        Kokkos::Impl::DeepCopy<HostSpace,ResCudaSpace>( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );

        //Formatting dependent on sizeof(uintptr_t)
        const char * format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "Cuda [ 0x%.12lx + %ld ] %s\n";
        }
        else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "Cuda [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf( buffer , 256
                , format_string
                , reinterpret_cast< uintptr_t >( r->data() )
                , r->size()
                , head.m_label
                );
      }
      else {
        snprintf( buffer , 256 , "Cuda [ 0 + 0 ]\n" );
      }
      s << buffer ;
      r = r->m_next ;
    } while ( r != & s_root_record );
  }
#else
  Kokkos::Impl::throw_runtime_exception("SharedAllocationHeader<ResCudaSpace>::print_records only works with KOKKOS_DEBUG enabled");
#endif
}


} // namespace Impl
} // namespace Kokkos
#else
void KOKKOS_CORE_SRC_CUDA_CUDASPACE_PREVENT_LINK_ERROR() {}
#endif // KOKKOS_ENABLE_CUDA

