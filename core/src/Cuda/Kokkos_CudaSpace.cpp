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

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <Kokkos_Macros.hpp>

/* only compile this file if CUDA is enabled for Kokkos */
#ifdef KOKKOS_HAVE_CUDA

#include <Kokkos_Cuda.hpp>
#include <Kokkos_CudaSpace.hpp>

#include <Cuda/Kokkos_Cuda_BasicAllocators.hpp>
#include <Cuda/Kokkos_Cuda_Internal.hpp>
#include <impl/Kokkos_Error.hpp>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

DeepCopy<CudaSpace,CudaSpace>::DeepCopy( void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpy( dst , src , n , cudaMemcpyDefault ) ); }

DeepCopy<CudaSpace,CudaSpace>::DeepCopy( const Cuda & instance , void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpyAsync( dst , src , n , cudaMemcpyDefault , instance.m_stream ) ); }

DeepCopy<HostSpace,CudaSpace>::DeepCopy( void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpy( dst , src , n , cudaMemcpyDefault ) ); }

DeepCopy<HostSpace,CudaSpace>::DeepCopy( const Cuda & instance , void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpyAsync( dst , src , n , cudaMemcpyDefault , instance.m_stream ) ); }

DeepCopy<CudaSpace,HostSpace>::DeepCopy( void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpy( dst , src , n , cudaMemcpyDefault ) ); }

DeepCopy<CudaSpace,HostSpace>::DeepCopy( const Cuda & instance , void * dst , const void * src , size_t n )
{ CUDA_SAFE_CALL( cudaMemcpyAsync( dst , src , n , cudaMemcpyDefault , instance.m_stream ) ); }

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/


namespace Kokkos {

namespace {

void texture_object_attach_impl(  Impl::AllocationTracker const & tracker
                                , unsigned type_size
                                , ::cudaChannelFormatDesc const & desc
                               )
{
  enum { TEXTURE_BOUND_1D = 2u << 27 };

  if ( tracker.attribute() == NULL ) {
    // check for correct allocator
    const bool ok_alloc =  tracker.allocator()->support_texture_binding();

    const bool ok_count = (tracker.alloc_size() / type_size) < TEXTURE_BOUND_1D;

    if (ok_alloc && ok_count) {
      Impl::TextureAttribute * attr = new Impl::TextureAttribute( tracker.alloc_ptr(), tracker.alloc_size(), desc );
      tracker.set_attribute( attr );
    }
    else {
      std::ostringstream oss;
      oss << "Error: Cannot attach texture object";
      if (!ok_alloc) {
        oss << ", incompatabile allocator " << tracker.allocator()->name();
      }
      if (!ok_count) {
        oss << ", array " << tracker.label() << " too large";
      }
      oss << ".";
      Kokkos::Impl::throw_runtime_exception( oss.str() );
    }
  }

  if ( NULL == dynamic_cast<Impl::TextureAttribute *>(tracker.attribute()) ) {
    std::ostringstream oss;
    oss << "Error: Allocation " << tracker.label() << " already has an attribute attached.";
    Kokkos::Impl::throw_runtime_exception( oss.str() );
  }

}

} // unnamed namespace

/*--------------------------------------------------------------------------*/

Impl::AllocationTracker CudaSpace::allocate_and_track( const std::string & label, const size_t size )
{
  return Impl::AllocationTracker( allocator(), size, label);
}

void CudaSpace::texture_object_attach(  Impl::AllocationTracker const & tracker
                                      , unsigned type_size
                                      , ::cudaChannelFormatDesc const & desc
                                     )
{
  texture_object_attach_impl( tracker, type_size, desc );
}

void CudaSpace::access_error()
{
  const std::string msg("Kokkos::CudaSpace::access_error attempt to execute Cuda function from non-Cuda space" );
  Kokkos::Impl::throw_runtime_exception( msg );
}

void CudaSpace::access_error( const void * const )
{
  const std::string msg("Kokkos::CudaSpace::access_error attempt to execute Cuda function from non-Cuda space" );
  Kokkos::Impl::throw_runtime_exception( msg );
}

/*--------------------------------------------------------------------------*/

Impl::AllocationTracker CudaUVMSpace::allocate_and_track( const std::string & label, const size_t size )
{
  return Impl::AllocationTracker( allocator(), size, label);
}

void CudaUVMSpace::texture_object_attach(  Impl::AllocationTracker const & tracker
                                         , unsigned type_size
                                         , ::cudaChannelFormatDesc const & desc
                                        )
{
  texture_object_attach_impl( tracker, type_size, desc );
}

bool CudaUVMSpace::available()
{
#if defined( CUDA_VERSION ) && ( 6000 <= CUDA_VERSION ) && !defined(__APPLE__)
  enum { UVM_available = true };
#else
  enum { UVM_available = false };
#endif
  return UVM_available;
}

/*--------------------------------------------------------------------------*/

Impl::AllocationTracker CudaHostPinnedSpace::allocate_and_track( const std::string & label, const size_t size )
{
  return Impl::AllocationTracker( allocator(), size, label);
}

/*--------------------------------------------------------------------------*/

} // namespace Kokkos


#endif // KOKKOS_HAVE_CUDA

