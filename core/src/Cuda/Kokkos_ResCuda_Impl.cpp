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

/*--------------------------------------------------------------------------*/
/* Kokkos interfaces */

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <Kokkos_Core.hpp>

#include <Cuda/Kokkos_Cuda_Error.hpp>
#include <Cuda/Kokkos_Cuda_Instance.hpp>
#include <Cuda/Kokkos_Cuda_Locks.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <cstdlib>

/* Standard 'C++' libraries */
#include <vector>
#include <iostream>
#include <sstream>
#include <string>


/*--------------------------------------------------------------------------*/


//----------------------------------------------------------------------------

namespace Kokkos {

ResCuda::size_type ResCuda::detect_device_count()
{ return Cuda::detect_device_count();}

int ResCuda::concurrency()
{ return Cuda::concurrency();}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
   int ResCuda::is_initialized()
   { return Kokkos::Impl::CudaInternal::singleton().is_initialized(); }

   void ResCuda::finalize()
   {
      Kokkos::Impl::CudaInternal::singleton().finalize();

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::finalize();
      #endif
   }

   void ResCuda::initialize( const Cuda::SelectDevice config , size_t num_instances )
   {
     Kokkos::Impl::CudaInternal::singleton().initialize( config , 0 );

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::initialize();
      #endif
   }
#else
  //! Has been initialized
  int ResCuda::impl_is_initialized() {
     return Kokkos::Impl::CudaInternal::singleton().is_initialized();
  }

  void ResCuda::impl_finalize() {
      Kokkos::Impl::CudaInternal::singleton().finalize();

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::finalize();
      #endif
  }

  void ResCuda::impl_initialize( const SelectDevice config, const size_t num_instances ) {
     Kokkos::Impl::CudaInternal::singleton().initialize( config.cuda_device_id , 0 );

      #if defined(KOKKOS_ENABLE_PROFILING)
        Kokkos::Profiling::initialize();
      #endif
  }
#endif


std::vector<unsigned>
ResCuda::detect_device_arch()
{
   return Cuda::detect_device_arch() ;
}

ResCuda::size_type ResCuda::device_arch()
{
  return Cuda::device_arch() ;
}


ResCuda::ResCuda()
  : Cuda() {
}

ResCuda::ResCuda( cudaStream_t stream )
  : Cuda( stream )
{}

void ResCuda::print_configuration( std::ostream & s , const bool )
{ Kokkos::Impl::CudaInternal::singleton().print_configuration( s ); }

void ResCuda::fence()
{
  Cuda::fence();
}

const char* ResCuda::name() { return "ResCuda"; }

} // namespace Kokkos

#else

void KOKKOS_CORE_SRC_CUDA_IMPL_PREVENT_LINK_ERROR() {}

#endif // KOKKOS_ENABLE_CUDA

