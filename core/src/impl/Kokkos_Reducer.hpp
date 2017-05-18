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

#include <impl/Kokkos_Traits.hpp>

//----------------------------------------------------------------------------
/*  Reducer abstraction:
 *  1) Provides 'join' operation
 *  2) Provides 'init' operation
 *  3) Optionally provides result value in a memory space
 *
 *  Created from:
 *  1) Functor::operator()( destination , source )
 *  2) Functor::{ join , init )
 */
//----------------------------------------------------------------------------

namespace Kokkos2 {
namespace Impl {

//----------------------------------------------------------------------------

template< typename value_type
        , bool = std::is_arithmetic< value_type >::value >
struct ReduceSum
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = value_type(0); }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept { dest += src ; }
};

template< typename value_type
        , bool = std::is_arithmetic< value_type >::value >
struct ReduceProd
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = value_type(1); }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept { dest *= src ; }
};

template< typename value_type
        , bool = std::is_integral< value_type >::value >
struct ReduceLAnd
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = 1 ; }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { dest = dest && src ; }
};

template< typename value_type
        , bool = std::is_integral< value_type >::value >
struct ReduceLOr
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = 0 ; }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { dest = dest || src ; }
};

template< typename value_type
        , bool = std::is_integral< value_type >::value >
struct ReduceLXor
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = 0 ; }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { dest = dest ? ( ! src ) : src ; }
};

template< typename value_type
        , bool = std::is_integral< value_type >::value >
struct ReduceBAnd
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = ~value_type(0); }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { dest = dest & src ; }
};

template< typename value_type
        , bool = std::is_integral< value_type >::value >
struct ReduceBOr
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = value_type(0); }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { dest = dest | src ; }
};

template< typename value_type
        , bool = std::is_integral< value_type >::value >
struct ReduceBXor
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept { dest = value_type(0); }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { dest = dest ^ src ; }
};

template< typename value_type
        , bool = std::is_arithmetic< value_type >::value >
struct ReduceMin
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept
    { dest = std::numeric_limits<value_type>::max(); }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { if ( src < dest ) dest = src ; }
};

template< typename value_type
        , bool = std::is_arithmetic< value_type >::value >
struct ReduceMax
{
  KOKKOS_INLINE_FUNCTION static
  void init( value_type & dest ) noexcept
    { dest = std::numeric_limits<value_type>::min(); }

  KOKKOS_INLINE_FUNCTION static
  void join( value_type volatile & dest
           , value_type const volatile & src ) noexcept
    { if ( dest < src ) dest = src ; }
};

template< typename pair_type
        , bool = std::is_arithmetic< typename pair_type::first_type >::value
              && std::is_integral< typename pair_type::second_type >::value
        >
struct ReduceMinLoc
{
  KOKKOS_INLINE_FUNCTION static
  void init( pair_type & dest ) noexcept
    {
      dest.first  = std::numeric_limits<typename pair_type::first_type >::max();
      dest.second = std::numeric_limits<typename pair_type::second_type>::max();
    }

  KOKKOS_INLINE_FUNCTION static
  void join( pair_type volatile & dest
           , pair_type const volatile & src ) noexcept
    { if ( src.first < dest.first ) dest = src ; }
};

template< typename pair_type
        , bool = std::is_arithmetic< typename pair_type::first_type >::value
              && std::is_integral< typename pair_type::second_type >::value
        >
struct ReduceMaxLoc
{
  KOKKOS_INLINE_FUNCTION static
  void init( pair_type & dest ) noexcept
    {
      dest.first  = std::numeric_limits<typename pair_type::first_type >::min();
      dest.second = std::numeric_limits<typename pair_type::second_type>::max();
    }

  KOKKOS_INLINE_FUNCTION static
  void join( pair_type volatile & dest
           , pair_type const volatile & src ) noexcept
    { if ( dest.first < src.first ) dest = src ; }
};

template< typename pair_type
        , bool = std::is_arithmetic< typename pair_type::first_type >::value
              && std::is_arithmetic< typename pair_type::second_type >::value
        >
struct ReduceMinMax
{
  KOKKOS_INLINE_FUNCTION static
  void init( pair_type & dest ) noexcept
    {
      dest.first  = std::numeric_limits< typename pair_type::first_type >::max();
      dest.second = std::numeric_limits< typename pair_type::second_type >::min();
    }

  KOKKOS_INLINE_FUNCTION static
  void join( pair_type volatile & dest
           , pair_type const volatile & src ) noexcept
    {
      if ( src.first < dest.first ) dest.first = src.first ;
      if ( dest.second < src.second ) dest.second = src.second ;
    }
};

//----------------------------------------------------------------------------

template< typename ValueType
        , class ReduceOp = ReduceSum< ValueType >
        , class MemorySpace = void
        , int Rank = std::rank<ValueType>::value
        >
struct Reducer ;


template< typename ValueType , class ReduceOp , class MemorySpace >
struct Reducer< ValueType , ReduceOp , MemorySpace , 0 >
  : private ReduceOp
{
public:

  using reducer        = Reducer ;
  using value_type     = ValueType ;
  using memory_space   = MemorySpace ;
  using reference_type = value_type & ;

private:

  //--------------------------------------------------------------------------
  // Determine what functions 'ReduceOp' provides:
  //   init( destination )
  //   join( destination , source )
  //
  //   operator()( destination , source )
  //
  // Provide defaults for missing optional operations

  template< class R , typename = void >
  struct INIT {
    KOKKOS_INLINE_FUNCTION static
    void init( R const & , value_type * dst ) { new(dst) value_type(); }
  };

  template< class R >
  struct INIT< R , decltype( ((R*)0)->init( *((value_type*)0 ) ) ) >
  {
    KOKKOS_INLINE_FUNCTION static
    void init( R const & r , value_type * dst ) { r.init( *dst ); }
  };

  template< class R , typename V , typename = void > struct JOIN
    {
      // If no join function then try operator()
      KOKKOS_INLINE_FUNCTION static
      void join( R const & r , V * dst , V const * src )
        { r.operator()(*dst,*src); }
    };

  template< class R , typename V >
  struct JOIN< R , V , decltype( ((R*)0)->join ( *((V *)0) , *((V const *)0) ) ) >
    {
      // If has join function use it
      KOKKOS_INLINE_FUNCTION static
      void join( R const & r , V * dst , V const * src )
        { r.join(*dst,*src); }
    };

  //--------------------------------------------------------------------------

  value_type * const m_result ;

public:

  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION constexpr static
  int length() noexcept { return 1 ; }

  KOKKOS_INLINE_FUNCTION constexpr
  value_type * data() const noexcept { return m_result ; }

  KOKKOS_INLINE_FUNCTION constexpr
  reference_type reference() const noexcept { return *m_result ; }

  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION
  void copy( value_type * const dest
           , value_type const * const src ) const noexcept
    { *dest = *src ; }

  KOKKOS_INLINE_FUNCTION
  void join( value_type * const dest
           , value_type const * const src ) const noexcept
    {
      Reducer::template JOIN<ReduceOp,value_type>::
        join( (ReduceOp const &) *this , dest , src );
    }

  KOKKOS_INLINE_FUNCTION
  void join( value_type volatile * const dest
           , value_type volatile const * const src ) const noexcept
    {
      Reducer::template JOIN<ReduceOp,value_type volatile>::
        join( (ReduceOp const &) *this , dest , src );
    }

  KOKKOS_INLINE_FUNCTION
  void init( value_type * dest ) const noexcept
    {
      Reducer::template INIT<ReduceOp>::
        init( (ReduceOp const &) *this , dest );
    }

  KOKKOS_INLINE_FUNCTION
  void final( value_type * ) const noexcept {}

  //--------------------------------------------------------------------------

  Reducer( Reducer const & ) = default ;
  Reducer( Reducer && ) = default ;
  Reducer & operator = ( Reducer const & ) = delete ;
  Reducer & operator = ( Reducer && ) = delete ;

  template< class Space >
  using rebind = Reducer< ValueType , ReduceOp , Space , 0 > ;

  template< class Space >
  KOKKOS_INLINE_FUNCTION constexpr
  Reducer( const Reducer< ValueType , ReduceOp , Space , 0 > & arg
         , value_type * arg_value
         ) noexcept
    : ReduceOp( arg ), m_result( arg_value ) {}

  template< typename ArgT >
  KOKKOS_INLINE_FUNCTION explicit
  constexpr Reducer
    ( ArgT * arg_value
    , typename std::enable_if
        < std::is_same<ArgT,value_type>::value &&
          std::is_default_constructible< ReduceOp >::value
        >::type * = 0
    ) noexcept
    : ReduceOp(), m_result( arg_value ) {}

  KOKKOS_INLINE_FUNCTION explicit
  constexpr Reducer( ReduceOp const & arg_op
                   , value_type     * arg_value = 0 ) noexcept
    : ReduceOp( arg_op ), m_result( arg_value ) {}

  KOKKOS_INLINE_FUNCTION explicit
  constexpr Reducer( ReduceOp      && arg_op
                   , value_type     * arg_value = 0 ) noexcept
    : ReduceOp( arg_op ), m_result( arg_value ) {}
};

//----------------------------------------------------------------------------
// Given a runtime array type and scalar reduce operator
// generate a reducer.

template< typename ValueType , class ReduceOp , class MemorySpace >
struct Reducer< ValueType , ReduceOp , MemorySpace , 1 >
  : private ReduceOp
{
public:

  using reducer        = Reducer ;
  using value_type     = typename std::remove_extent<ValueType>::type ;
  using memory_space   = MemorySpace ;
  using reference_type = value_type * ;

  static_assert( std::extent<ValueType>::value == 0
               , "Kokkos::Impl::Reducer array ValueType must be T[]" );

private:

  //--------------------------------------------------------------------------
  // Determine what functions 'ReduceOp' provides:
  //   init( destination )
  //   join( destination , source )
  //   operator()( destination , source )
  //
  // Provide defaults for missing optional operations

  template< class R , typename = void >
  struct INIT {
    KOKKOS_INLINE_FUNCTION static
    void init( R const & , value_type * dst ) { new(dst) value_type(); }
  };

  template< class R >
  struct INIT< R , decltype( ((R*)0)->init( *((value_type*)0 ) ) ) >
  {
    KOKKOS_INLINE_FUNCTION static
    void init( R const & r , value_type * dst ) { r.init( *dst ); }
  };

  template< class R , typename V , typename = void > struct JOIN
    {
      // If no join function then try operator()
      KOKKOS_INLINE_FUNCTION static
      void join( R const & r , V * dst , V const * src )
        { r.operator()(*dst,*src); }
    };

  template< class R , typename V >
  struct JOIN< R , V , decltype( ((R*)0)->join ( *((V *)0) , *((V const *)0) ) ) >
    {
      // If has join function use it
      KOKKOS_INLINE_FUNCTION static
      void join( R const & r , V * dst , V const * src )
        { r.join(*dst,*src); }
    };

  //--------------------------------------------------------------------------

  value_type * const m_result ;
  int const          m_count ;

public:


  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION constexpr
  int length() const noexcept { return m_count ; }

  KOKKOS_INLINE_FUNCTION constexpr
  value_type * data() const noexcept { return m_result ; }

  KOKKOS_INLINE_FUNCTION constexpr
  reference_type reference() const noexcept { return m_result ; }

  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION
  void copy( value_type * const dest
           , value_type const * const src ) const noexcept
    { for ( int i = 0 ; i < length() ; ++i ) { dest[i] = src[i] ; } }

  KOKKOS_INLINE_FUNCTION
  void join( value_type * const dest
           , value_type const * const src ) const noexcept
    {
      for ( int i = 0 ; i < length() ; ++i ) {
        Reducer::template JOIN<ReduceOp,value_type>::join( (ReduceOp &) *this , dest[i] , src[i] );
      }
    }

  KOKKOS_INLINE_FUNCTION
  void join( value_type volatile * const dest
           , value_type volatile const * const src ) const noexcept
    {
      for ( int i = 0 ; i < length() ; ++i ) {
        Reducer::template JOIN<ReduceOp,value_type volatile>::join( (ReduceOp &) *this , dest[i] , src[i] );
      }
    }

  KOKKOS_INLINE_FUNCTION
  void init( value_type * dest ) const noexcept
    {
      for ( int i = 0 ; i < length() ; ++i ) {
        Reducer::template INIT<ReduceOp>::init( (ReduceOp &) *this , dest[i] );
      }
    }

  KOKKOS_INLINE_FUNCTION
  void final( value_type * ) const noexcept {}

  //--------------------------------------------------------------------------

  Reducer( Reducer const & ) = default ;
  Reducer( Reducer && ) = default ;
  Reducer & operator = ( Reducer const & ) = delete ;
  Reducer & operator = ( Reducer && ) = delete ;

  template< class Space >
  using rebind = Reducer< ValueType , ReduceOp , Space , 1 > ;

  template< class Space >
  KOKKOS_INLINE_FUNCTION constexpr
  Reducer( Reducer< ValueType , ReduceOp , Space , 1 > const & arg
         , value_type * arg_value
         ) noexcept
    : ReduceOp( arg ), m_result( arg_value ), m_count( arg.length() ) {}

  template< typename ArgT >
  KOKKOS_INLINE_FUNCTION constexpr
  Reducer( ArgT * arg_value
         , typename std::enable_if
             < std::is_same<ArgT,value_type>::value &&
               std::is_default_constructible< ReduceOp >::value
             , unsigned >::type arg_length
         ) noexcept
         : ReduceOp(), m_result( arg_value ), m_count(arg_length) {}

  KOKKOS_INLINE_FUNCTION explicit
  constexpr Reducer( ReduceOp const & arg_op
                   , value_type     * arg_value
                   , int arg_length ) noexcept
    : ReduceOp( arg_op ) , m_result( arg_value ), m_count( arg_length ) {}

  KOKKOS_INLINE_FUNCTION explicit
  constexpr Reducer( ReduceOp      && arg_op
                   , value_type     * arg_value
                   , int arg_length ) noexcept
    : ReduceOp( arg_op ) , m_result( arg_value ), m_count( arg_length ) {}
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos2 {

template< typename ValueType >
constexpr
Impl::Reducer< ValueType , Impl::ReduceSum< ValueType > >
Sum( ValueType & arg_value )
{
  return Impl::Reducer< ValueType , Impl::ReduceSum< ValueType > >( & arg_value );
}

template< typename ValueType >
constexpr
Impl::Reducer< ValueType , Impl::ReduceMin< ValueType > >
Min( ValueType & arg_value )
{
  return Impl::Reducer< ValueType , Impl::ReduceMin< ValueType > >( & arg_value );
}

template< typename ValueType >
constexpr
Impl::Reducer< ValueType , Impl::ReduceMax< ValueType > >
Max( ValueType & arg_value )
{
  return Impl::Reducer< ValueType , Impl::ReduceMax< ValueType > >( & arg_value );
}

template< typename ValueType >
constexpr
Impl::Reducer< ValueType[] , Impl::ReduceSum< ValueType > >
Sum( ValueType * arg_value , int arg_length )
{
  return Impl::Reducer< ValueType[] , Impl::ReduceSum< ValueType > >( arg_value , arg_length );
}

//----------------------------------------------------------------------------

template< typename ValueType , class JoinType >
Impl::Reducer< ValueType , JoinType >
reducer( ValueType & value , JoinType const & lambda )
{
  return Impl::Reducer< ValueType , JoinType >( lambda , & value );
}

//----------------------------------------------------------------------------

} // namespace Kokkos

#endif /* #ifndef KOKKOS_IMPL_REDUCER_HPP */

