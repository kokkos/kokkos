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

#ifndef KOKKOS_CORE_IMPL_UTILITIES_HPP
#define KOKKOS_CORE_IMPL_UTILITIES_HPP

#include <Kokkos_Macros.hpp>
#include <type_traits>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos { namespace Impl {

// same as std::forward
// needed to allow perfect forwarding on the device
template <typename T>
KOKKOS_INLINE_FUNCTION
constexpr
T&& forward( typename std::remove_reference<T>::type& arg ) noexcept
{ return static_cast<T&&>(arg); }

template <typename T>
KOKKOS_INLINE_FUNCTION
constexpr
T&& forward( typename std::remove_reference<T>::type&& arg ) noexcept
{ return static_cast<T&&>(arg); }

// same as std::move
// needed to allowing moving on the device
template <typename T>
KOKKOS_INLINE_FUNCTION
constexpr
typename std::remove_reference<T>::type&& move( T&& arg ) noexcept
{ return static_cast<typename std::remove_reference<T>::type&&>(arg); }

// empty function to allow expanding a variadic argument pack
template<typename... Args>
KOKKOS_INLINE_FUNCTION
void expand_variadic(Args &&...) {}

//----------------------------------------
// C++14 integer sequence
template< typename T , T ... Ints >
struct integer_sequence {
  using value_type = T ;
  static constexpr std::size_t size() { return sizeof...(Ints); }
};

template< typename T , std::size_t N >
struct make_integer_sequence_helper ;

template< typename T , T N >
using make_integer_sequence =
  typename make_integer_sequence_helper<T,N>::type ;


template< typename T >
struct make_integer_sequence_helper< T , 0 >
{ using type = integer_sequence<T> ; };

template< typename T >
struct make_integer_sequence_helper< T , 1 >
{ using type = integer_sequence<T,0> ; };

template< typename T >
struct make_integer_sequence_helper< T , 2 >
{ using type = integer_sequence<T,0,1> ; };

template< typename T >
struct make_integer_sequence_helper< T , 3 >
{ using type = integer_sequence<T,0,1,2> ; };

template< typename T >
struct make_integer_sequence_helper< T , 4 >
{ using type = integer_sequence<T,0,1,2,3> ; };

template< typename T >
struct make_integer_sequence_helper< T , 5 >
{ using type = integer_sequence<T,0,1,2,3,4> ; };

template< typename T >
struct make_integer_sequence_helper< T , 6 >
{ using type = integer_sequence<T,0,1,2,3,4,5> ; };

template< typename T >
struct make_integer_sequence_helper< T , 7 >
{ using type = integer_sequence<T,0,1,2,3,4,5,6> ; };

template< typename T >
struct make_integer_sequence_helper< T , 8 >
{ using type = integer_sequence<T,0,1,2,3,4,5,6,7> ; };

template< typename X , typename Y >
struct make_integer_sequence_concat ;

template< typename T , T ... x , T ... y >
struct make_integer_sequence_concat< integer_sequence<T,x...>
                                   , integer_sequence<T,y...> >
{ using type = integer_sequence< T , x ... , (sizeof...(x)+y)... > ; };

template< typename T , std::size_t N >
struct make_integer_sequence_helper {
  using type = typename make_integer_sequence_concat
    < typename make_integer_sequence_helper< T , N/2 >::type
    , typename make_integer_sequence_helper< T , N - N/2 >::type
    >::type ;
};

//----------------------------------------

}} // namespace Kokkos::Impl


#endif //KOKKOS_CORE_IMPL_UTILITIES
