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

#ifndef KOKKOS_TAGS_HPP
#define KOKKOS_TAGS_HPP

#include <impl/Kokkos_Traits.hpp>
#include <Kokkos_Core_fwd.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class C , class Enable = void >
struct is_memory_space_enable
{ typedef std::false_type type ; };

template< class C >
struct is_memory_space_enable< C ,
  typename std::enable_if<
    std::is_same< C , typename C::memory_space >::value
  >::type >
{ typedef std::true_type type ; };


template< class C , class Enable = void >
struct is_execution_space_enable
{ typedef std::false_type type ; };

template< class C >
struct is_execution_space_enable< C ,
  typename std::enable_if<
    std::is_same< C , typename C::execution_space >::value
  >::type >
{ typedef std::true_type type ; };


template< class C , class Enable = void >
struct is_execution_policy_enable
{ typedef std::false_type type ; };

template< class C >
struct is_execution_policy_enable< C ,
  typename std::enable_if<
    std::is_same< C , typename C::execution_policy >::value
  >::type >
{ typedef std::true_type type ; };


template< class C , class Enable = void >
struct is_array_layout_enable
{ typedef std::false_type type ; };

template< class C >
struct is_array_layout_enable< C ,
  typename std::enable_if<
    std::is_same< C , typename C::array_layout >::value
  >::type >
{ typedef std::true_type type ; };


template< class C , class Enable = void >
struct is_memory_traits_enable
{ typedef std::false_type type ; };

template< class C >
struct is_memory_traits_enable< C ,
  typename std::enable_if<
    std::is_same< C , typename C::memory_traits >::value
  >::type >
{ typedef std::true_type type ; };


template< class C >
using is_memory_space = typename is_memory_space_enable<C>::type ;

template< class C >
using is_execution_space = typename is_execution_space_enable<C>::type ;

template< class C >
using is_execution_policy = typename is_execution_policy_enable<C>::type ;

template< class C >
using is_array_layout = typename is_array_layout_enable<C>::type ;

template< class C >
using is_memory_traits = typename is_memory_traits_enable<C>::type ;

}
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template< class ExecutionSpace , class MemorySpace >
struct Device {
  static_assert( Impl::is_execution_space<ExecutionSpace>::value
               , "Execution space is not valid" );
  static_assert( Impl::is_memory_space<MemorySpace>::value
               , "Memory space is not valid" );
  typedef ExecutionSpace execution_space;
  typedef MemorySpace memory_space;
  typedef Device<execution_space,memory_space> device_type;
};
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< class C , class Enable = void >
struct is_space : public Impl::false_type {};

template< class C >
struct is_space< C
                 , typename Impl::enable_if<(
                     Impl::is_same< C , typename C::execution_space >::value ||
                     Impl::is_same< C , typename C::memory_space    >::value ||
                     Impl::is_same< C , Device<
                                             typename C::execution_space,
                                             typename C::memory_space> >::value
                   )>::type
                 >
  : public Impl::true_type
{
  typedef typename C::execution_space  execution_space ;
  typedef typename C::memory_space     memory_space ;

  // The host_memory_space defines a space with host-resident memory.
  // If the execution space's memory space is host accessible then use that execution space.
  // else use the HostSpace.
  typedef
      typename Impl::if_c< Impl::is_same< memory_space , HostSpace >::value
#ifdef KOKKOS_HAVE_CUDA
                        || Impl::is_same< memory_space , CudaUVMSpace>::value
                        || Impl::is_same< memory_space , CudaHostPinnedSpace>::value
#endif
                          , memory_space , HostSpace >::type
      host_memory_space ;

  // The host_execution_space defines a space which has access to HostSpace.
  // If the execution space can access HostSpace then use that execution space.
  // else use the DefaultHostExecutionSpace.
#ifdef KOKKOS_HAVE_CUDA
  typedef
      typename Impl::if_c< Impl::is_same< execution_space , Cuda >::value
                          , DefaultHostExecutionSpace , execution_space >::type
      host_execution_space ;
#else
  typedef execution_space host_execution_space;
#endif

  typedef Device<host_execution_space,host_memory_space> host_mirror_space;
};
}
}

#endif
