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

#ifndef KOKKOS_IMPL_REDUCER_HPP
#define KOKKOS_IMPL_REDUCER_HPP

namespace Kokkos {
namespace Impl {

template< typename value_type >
struct ReduceSum
{
  KOKKOS_INLINE_FUNCTION static
  void join( value_type & dest
           , value_type const & src ) noexcept
    { dest = src ; }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { dest += src ; }

  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept
    { dest = value_type(0); }
};

template< typename T
        , template<typename> class ReduceOp = ReduceSum
        , typename MemorySpace = void >
struct Reducer
{
private:

  enum : int { rank = std::rank<T>::value };

public:

  using reducer      = Reducer ;
  using memory_space = MemorySpace ;
  using value_type   = typename std::remove_extent<T>::type ;
  using reduce_op    = ReduceOp< value_type > ;
  using reference_type =
    typename std::conditional< ( rank != 0 )
                             , value_type *
                             , value_type &
                             >::type ;

  KOKKOS_INLINE_FUNCTION
  void join( value_type & dest
           , value_type const & src ) const noexcept
    { reduce_op::join( dest , src ); }

  KOKKOS_INLINE_FUNCTION
  void join( value_type volatile & dest
           , value_type const volatile & src ) const noexcept
    { reduce_op::join( dest , src ); }

  KOKKOS_INLINE_FUNCTION
  void init( value_type & dest ) const noexcept
    { reduce_op::init( dest ); }


  KOKKOS_INLINE_FUNCTION
  void join( value_type * const dest
           , value_type const * const src ) const noexcept
    {
      for ( int i = 0 ; i < m_length ; ++i ) {
        reduce_op::join( dest[i] , src[i] );
      }
    }

  KOKKOS_INLINE_FUNCTION
  void join( value_type volatile * const dest
           , value_type const volatile * const src ) const noexcept
    {
      for ( int i = 0 ; i < m_length ; ++i ) {
        reduce_op::join( dest[i] , src[i] );
      }
    }

  KOKKOS_INLINE_FUNCTION
  void init( value_type * dest ) const noexcept
    { for ( int i = 0 ; i < m_length ; ++i ) reduce_op::init( dest[i] ); }

  KOKKOS_INLINE_FUNCTION
  constexpr Reducer() noexcept
    : m_result(0), m_length(0) {}

  KOKKOS_INLINE_FUNCTION explicit
  constexpr Reducer( value_type & arg ) noexcept
    : m_result( & arg ), m_length(1) {}

  KOKKOS_INLINE_FUNCTION
  constexpr Reducer( value_type * const arg_value , int arg_length ) noexcept
    : m_result( arg_value ), m_length( arg_length ) {}

  KOKKOS_INLINE_FUNCTION
  constexpr Reducer( int arg_length ) noexcept
    : m_result(0), m_length( arg_length ) {}

  KOKKOS_INLINE_FUNCTION
  constexpr int length() const noexcept { return m_length ; }

  KOKKOS_INLINE_FUNCTION
  value_type & operator[]( int i ) const noexcept
    { return m_result[i]; }

private:

  template< int Rank >
  static constexpr
  typename std::enable_if< ( 0 != Rank ) , reference_type >::type
  ref( value_type * p ) noexcept { return p ; }

  template< int Rank >
  static constexpr
  typename std::enable_if< ( 0 == Rank ) , reference_type >::type
  ref( value_type * p ) noexcept { return *p ; }

public:

  KOKKOS_INLINE_FUNCTION
  reference_type result() const noexcept
    { return Reducer::template ref< rank >( m_result ); }

private:

  value_type * const m_result ;
  int          const m_length ;
};

} // namespace Impl

template< typename ValueType >
constexpr
Impl::Reducer< ValueType , Impl::ReduceSum >
Sum( ValueType & arg_value )
{
  static_assert( std::is_trivial<ValueType>::value
               , "Kokkos reducer requires trivial value type" );
  return Impl::Reducer< ValueType , Impl::ReduceSum >( arg_value );
}

template< typename ValueType >
constexpr
Impl::Reducer< ValueType[] , Impl::ReduceSum >
Sum( ValueType * arg_value , int arg_length )
{
  static_assert( std::is_trivial<ValueType>::value
               , "Kokkos reducer requires trivial value type" );
  return Impl::Reducer< ValueType[] , Impl::ReduceSum >( arg_value , arg_length );
}

} // namespace Kokkos

#endif /* #ifndef KOKKOS_IMPL_REDUCER_HPP */

