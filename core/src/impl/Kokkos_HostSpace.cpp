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
/*--------------------------------------------------------------------------*/

