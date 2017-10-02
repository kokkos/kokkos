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

#ifndef KOKKOS_EXPERIMENTAL_VIEW_MAPPING_HPP
#define KOKKOS_EXPERIMENTAL_VIEW_MAPPING_HPP

#include <type_traits>
#include <initializer_list>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Layout.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_ViewCtor.hpp>
#include <impl/Kokkos_Atomic_View.hpp>
#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< unsigned I , size_t ... Args >
struct variadic_size_t
  { enum { value = ~size_t(0) }; };

template< size_t Val , size_t ... Args >
struct variadic_size_t< 0 , Val , Args ... >
  { enum { value = Val }; };

template< unsigned I , size_t Val , size_t ... Args >
struct variadic_size_t< I , Val , Args ... >
  { enum { value = variadic_size_t< I - 1 , Args ... >::value }; };

template< size_t ... Args >
struct rank_dynamic ;

template<>
struct rank_dynamic<> { enum { value = 0 }; };

template< size_t Val , size_t ... Args >
struct rank_dynamic< Val , Args... >
{
  enum { value = ( Val == 0 ? 1 : 0 ) + rank_dynamic< Args... >::value };
};

#define KOKKOS_IMPL_VIEW_DIMENSION( R ) \
  template< size_t V , unsigned > struct ViewDimension ## R \
    { \
      enum { ArgN ## R = ( V != ~size_t(0) ? V : 1 ) }; \
      enum { N ## R = ( V != ~size_t(0) ? V : 1 ) }; \
      KOKKOS_INLINE_FUNCTION explicit ViewDimension ## R ( size_t ) {} \
      ViewDimension ## R () = default ; \
      ViewDimension ## R ( const ViewDimension ## R  & ) = default ; \
      ViewDimension ## R & operator = ( const ViewDimension ## R  & ) = default ; \
    }; \
  template< unsigned RD > struct ViewDimension ## R < 0 , RD > \
    { \
      enum { ArgN ## R = 0 }; \
      typename std::conditional<( RD < 3 ), size_t , unsigned >::type N ## R ; \
      ViewDimension ## R () = default ; \
      ViewDimension ## R ( const ViewDimension ## R  & ) = default ; \
      ViewDimension ## R & operator = ( const ViewDimension ## R  & ) = default ; \
      KOKKOS_INLINE_FUNCTION explicit ViewDimension ## R ( size_t V ) : N ## R ( V ) {} \
    };

KOKKOS_IMPL_VIEW_DIMENSION( 0 )
KOKKOS_IMPL_VIEW_DIMENSION( 1 )
KOKKOS_IMPL_VIEW_DIMENSION( 2 )
KOKKOS_IMPL_VIEW_DIMENSION( 3 )
KOKKOS_IMPL_VIEW_DIMENSION( 4 )
KOKKOS_IMPL_VIEW_DIMENSION( 5 )
KOKKOS_IMPL_VIEW_DIMENSION( 6 )
KOKKOS_IMPL_VIEW_DIMENSION( 7 )

#undef KOKKOS_IMPL_VIEW_DIMENSION

template< size_t ... Vals >
struct ViewDimension
  : public ViewDimension0< variadic_size_t<0,Vals...>::value
                         , rank_dynamic< Vals... >::value >
  , public ViewDimension1< variadic_size_t<1,Vals...>::value
                         , rank_dynamic< Vals... >::value >
  , public ViewDimension2< variadic_size_t<2,Vals...>::value
                         , rank_dynamic< Vals... >::value >
  , public ViewDimension3< variadic_size_t<3,Vals...>::value
                         , rank_dynamic< Vals... >::value >
  , public ViewDimension4< variadic_size_t<4,Vals...>::value
                         , rank_dynamic< Vals... >::value >
  , public ViewDimension5< variadic_size_t<5,Vals...>::value
                         , rank_dynamic< Vals... >::value >
  , public ViewDimension6< variadic_size_t<6,Vals...>::value
                         , rank_dynamic< Vals... >::value >
  , public ViewDimension7< variadic_size_t<7,Vals...>::value
                         , rank_dynamic< Vals... >::value >
{
  typedef ViewDimension0< variadic_size_t<0,Vals...>::value
                        , rank_dynamic< Vals... >::value > D0 ;
  typedef ViewDimension1< variadic_size_t<1,Vals...>::value
                        , rank_dynamic< Vals... >::value > D1 ;
  typedef ViewDimension2< variadic_size_t<2,Vals...>::value
                        , rank_dynamic< Vals... >::value > D2 ;
  typedef ViewDimension3< variadic_size_t<3,Vals...>::value
                        , rank_dynamic< Vals... >::value > D3 ;
  typedef ViewDimension4< variadic_size_t<4,Vals...>::value
                        , rank_dynamic< Vals... >::value > D4 ;
  typedef ViewDimension5< variadic_size_t<5,Vals...>::value
                        , rank_dynamic< Vals... >::value > D5 ;
  typedef ViewDimension6< variadic_size_t<6,Vals...>::value
                        , rank_dynamic< Vals... >::value > D6 ;
  typedef ViewDimension7< variadic_size_t<7,Vals...>::value
                        , rank_dynamic< Vals... >::value > D7 ;

  using D0::ArgN0 ;
  using D1::ArgN1 ;
  using D2::ArgN2 ;
  using D3::ArgN3 ;
  using D4::ArgN4 ;
  using D5::ArgN5 ;
  using D6::ArgN6 ;
  using D7::ArgN7 ;

  using D0::N0 ;
  using D1::N1 ;
  using D2::N2 ;
  using D3::N3 ;
  using D4::N4 ;
  using D5::N5 ;
  using D6::N6 ;
  using D7::N7 ;

  enum { rank = sizeof...(Vals) };
  enum { rank_dynamic = Impl::rank_dynamic< Vals... >::value };

  ViewDimension() = default ;
  ViewDimension( const ViewDimension & ) = default ;
  ViewDimension & operator = ( const ViewDimension & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr
  ViewDimension( size_t n0 , size_t n1 , size_t n2 , size_t n3
               , size_t n4 , size_t n5 , size_t n6 , size_t n7 )
    : D0( n0 )
    , D1( n1 )
    , D2( n2 )
    , D3( n3 )
    , D4( n4 )
    , D5( n5 )
    , D6( n6 )
    , D7( n7 )
    {}

  KOKKOS_INLINE_FUNCTION
  constexpr size_t extent( const unsigned r ) const
    {
      return r == 0 ? N0 : (
             r == 1 ? N1 : (
             r == 2 ? N2 : (
             r == 3 ? N3 : (
             r == 4 ? N4 : (
             r == 5 ? N5 : (
             r == 6 ? N6 : (
             r == 7 ? N7 : 0 )))))));
    }

  template< size_t N >
  struct prepend { typedef ViewDimension< N , Vals... > type ; };

  template< size_t N >
  struct append { typedef ViewDimension< Vals... , N > type ; };
};

template< class A , class B >
struct ViewDimensionJoin ;

template< size_t ... A , size_t ... B >
struct ViewDimensionJoin< ViewDimension< A... > , ViewDimension< B... > > {
  typedef ViewDimension< A... , B... > type ;
};

//----------------------------------------------------------------------------

template< class DstDim , class SrcDim >
struct ViewDimensionAssignable ;

template< size_t ... DstArgs , size_t ... SrcArgs >
struct ViewDimensionAssignable< ViewDimension< DstArgs ... >
                              , ViewDimension< SrcArgs ... > >
{
  typedef ViewDimension< DstArgs... > dst ;
  typedef ViewDimension< SrcArgs... > src ;

  enum { value =
    unsigned(dst::rank) == unsigned(src::rank) && (
      //Compile time check that potential static dimensions match
      ( ( 1 > dst::rank_dynamic && 1 > src::rank_dynamic ) ? (size_t(dst::ArgN0) == size_t(src::ArgN0)) : true ) &&
      ( ( 2 > dst::rank_dynamic && 2 > src::rank_dynamic ) ? (size_t(dst::ArgN1) == size_t(src::ArgN1)) : true ) &&
      ( ( 3 > dst::rank_dynamic && 3 > src::rank_dynamic ) ? (size_t(dst::ArgN2) == size_t(src::ArgN2)) : true ) &&
      ( ( 4 > dst::rank_dynamic && 4 > src::rank_dynamic ) ? (size_t(dst::ArgN3) == size_t(src::ArgN3)) : true ) &&
      ( ( 5 > dst::rank_dynamic && 5 > src::rank_dynamic ) ? (size_t(dst::ArgN4) == size_t(src::ArgN4)) : true ) &&
      ( ( 6 > dst::rank_dynamic && 6 > src::rank_dynamic ) ? (size_t(dst::ArgN5) == size_t(src::ArgN5)) : true ) &&
      ( ( 7 > dst::rank_dynamic && 7 > src::rank_dynamic ) ? (size_t(dst::ArgN6) == size_t(src::ArgN6)) : true ) &&
      ( ( 8 > dst::rank_dynamic && 8 > src::rank_dynamic ) ? (size_t(dst::ArgN7) == size_t(src::ArgN7)) : true )
    )};

};

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct ALL_t {
  KOKKOS_INLINE_FUNCTION
  constexpr const ALL_t & operator()() const { return *this ; }
};

}} // namespace Kokkos::Impl

namespace Kokkos {
namespace Impl {

template< class T >
struct is_integral_extent_type
{ enum { value = std::is_same<T,Kokkos::Impl::ALL_t>::value ? 1 : 0 }; };

template< class iType >
struct is_integral_extent_type< std::pair<iType,iType> >
{ enum { value = std::is_integral<iType>::value ? 1 : 0 }; };

template< class iType >
struct is_integral_extent_type< Kokkos::pair<iType,iType> >
{ enum { value = std::is_integral<iType>::value ? 1 : 0 }; };

// Assuming '2 == initializer_list<iType>::size()'
template< class iType >
struct is_integral_extent_type< std::initializer_list<iType> >
{ enum { value = std::is_integral<iType>::value ? 1 : 0 }; };

template < unsigned I , class ... Args >
struct is_integral_extent
{
  // get_type is void when sizeof...(Args) <= I
  typedef typename std::remove_cv<
          typename std::remove_reference<
          typename Kokkos::Impl::get_type<I,Args...
          >::type >::type >::type type ;

  enum { value = is_integral_extent_type<type>::value };

  static_assert( value ||
                 std::is_integral<type>::value ||
                 std::is_same<type,void>::value
               , "subview argument must be either integral or integral extent" );
};

// Rules for subview arguments and layouts matching

template<class LayoutDest, class LayoutSrc, int RankDest, int RankSrc, int CurrentArg, class ... SubViewArgs>
struct SubviewLegalArgsCompileTime;

// Rules which allow LayoutLeft to LayoutLeft assignment

template<int RankDest, int RankSrc, int CurrentArg, class Arg, class ... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::LayoutLeft, RankDest, RankSrc, CurrentArg, Arg, SubViewArgs...> {
  enum { value      =(((CurrentArg==RankDest-1) && (Kokkos::Impl::is_integral_extent_type<Arg>::value)) ||
                      ((CurrentArg>=RankDest) && (std::is_integral<Arg>::value)) ||
                      ((CurrentArg<RankDest) && (std::is_same<Arg,Kokkos::Impl::ALL_t>::value)) ||
                      ((CurrentArg==0) && (Kokkos::Impl::is_integral_extent_type<Arg>::value))
                     ) && (SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::LayoutLeft, RankDest, RankSrc, CurrentArg+1, SubViewArgs...>::value)};
};

template<int RankDest, int RankSrc, int CurrentArg, class Arg>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutLeft, Kokkos::LayoutLeft, RankDest, RankSrc, CurrentArg, Arg> {
  enum { value = ((CurrentArg==RankDest-1) || (std::is_integral<Arg>::value)) &&
                 (CurrentArg==RankSrc-1) };
};

// Rules which allow LayoutRight to LayoutRight assignment

template<int RankDest, int RankSrc, int CurrentArg, class Arg, class ... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutRight, Kokkos::LayoutRight, RankDest, RankSrc, CurrentArg, Arg, SubViewArgs...> {
  enum { value      =(((CurrentArg==RankSrc-RankDest) && (Kokkos::Impl::is_integral_extent_type<Arg>::value)) ||
                      ((CurrentArg<RankSrc-RankDest) && (std::is_integral<Arg>::value)) ||
                      ((CurrentArg>=RankSrc-RankDest) && (std::is_same<Arg,Kokkos::Impl::ALL_t>::value))
                     ) && (SubviewLegalArgsCompileTime<Kokkos::LayoutRight, Kokkos::LayoutRight, RankDest, RankSrc, CurrentArg+1, SubViewArgs...>::value)};
};

template<int RankDest, int RankSrc, int CurrentArg, class Arg>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutRight, Kokkos::LayoutRight, RankDest, RankSrc, CurrentArg, Arg> {
  enum { value = ((CurrentArg==RankSrc-1) && (std::is_same<Arg,Kokkos::Impl::ALL_t>::value)) };
};

// Rules which allow assignment to LayoutStride

template<int RankDest, int RankSrc, int CurrentArg, class ... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutStride,Kokkos::LayoutLeft,RankDest,RankSrc,CurrentArg,SubViewArgs...> {
  enum { value = true };
};

template<int RankDest, int RankSrc, int CurrentArg, class ... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutStride,Kokkos::LayoutRight,RankDest,RankSrc,CurrentArg,SubViewArgs...> {
  enum { value = true };
};

template<int RankDest, int RankSrc, int CurrentArg, class ... SubViewArgs>
struct SubviewLegalArgsCompileTime<Kokkos::LayoutStride,Kokkos::LayoutStride,RankDest,RankSrc,CurrentArg,SubViewArgs...> {
  enum { value = true };
};


template< unsigned DomainRank , unsigned RangeRank >
struct SubviewExtents {
private:

  // Cannot declare zero-length arrays
  enum { InternalRangeRank = RangeRank ? RangeRank : 1u };

  size_t   m_begin[  DomainRank ];
  size_t   m_length[ InternalRangeRank ];
  unsigned m_index[  InternalRangeRank ];

  template< size_t ... DimArgs >
  KOKKOS_FORCEINLINE_FUNCTION
  bool set( unsigned
          , unsigned
          , const ViewDimension< DimArgs ... > & )
    { return true ; }

  template< class T , size_t ... DimArgs , class ... Args >
  KOKKOS_FORCEINLINE_FUNCTION
  bool set( unsigned domain_rank
          , unsigned range_rank
          , const ViewDimension< DimArgs ... > & dim
          , const T & val
          , Args ... args )
    {
      const size_t v = static_cast<size_t>(val);

      m_begin[ domain_rank ] = v ;

      return set( domain_rank + 1 , range_rank , dim , args... )
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
             && ( v < dim.extent( domain_rank ) )
#endif
      ;
    }

  // ALL_t
  template< size_t ... DimArgs , class ... Args >
  KOKKOS_FORCEINLINE_FUNCTION
  bool set( unsigned domain_rank
          , unsigned range_rank
          , const ViewDimension< DimArgs ... > & dim
          , const Kokkos::Impl::ALL_t
          , Args ... args )
    {
      m_begin[  domain_rank ] = 0 ;
      m_length[ range_rank  ] = dim.extent( domain_rank );
      m_index[  range_rank  ] = domain_rank ;

      return set( domain_rank + 1 , range_rank + 1 , dim , args... );
    }

  // std::pair range
  template< class T , size_t ... DimArgs , class ... Args >
  KOKKOS_FORCEINLINE_FUNCTION
  bool set( unsigned domain_rank
          , unsigned range_rank
          , const ViewDimension< DimArgs ... > & dim
          , const std::pair<T,T> & val
          , Args ... args )
    {
      const size_t b = static_cast<size_t>( val.first );
      const size_t e = static_cast<size_t>( val.second );

      m_begin[  domain_rank ] = b ;
      m_length[ range_rank  ] = e - b ;
      m_index[  range_rank  ] = domain_rank ;

      return set( domain_rank + 1 , range_rank + 1 , dim , args... )
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
             && ( e <= b + dim.extent( domain_rank ) )
#endif
      ;
    }

  // Kokkos::pair range
  template< class T , size_t ... DimArgs , class ... Args >
  KOKKOS_FORCEINLINE_FUNCTION
  bool set( unsigned domain_rank
          , unsigned range_rank
          , const ViewDimension< DimArgs ... > & dim
          , const Kokkos::pair<T,T> & val
          , Args ... args )
    {
      const size_t b = static_cast<size_t>( val.first );
      const size_t e = static_cast<size_t>( val.second );

      m_begin[  domain_rank ] = b ;
      m_length[ range_rank  ] = e - b ;
      m_index[  range_rank  ] = domain_rank ;

      return set( domain_rank + 1 , range_rank + 1 , dim , args... )
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
             && ( e <= b + dim.extent( domain_rank ) )
#endif
      ;
    }

  // { begin , end } range
  template< class T , size_t ... DimArgs , class ... Args >
  KOKKOS_FORCEINLINE_FUNCTION
  bool set( unsigned domain_rank
          , unsigned range_rank
          , const ViewDimension< DimArgs ... > & dim
          , const std::initializer_list< T > & val
          , Args ... args )
    {
      const size_t b = static_cast<size_t>( val.begin()[0] );
      const size_t e = static_cast<size_t>( val.begin()[1] );

      m_begin[  domain_rank ] = b ;
      m_length[ range_rank  ] = e - b ;
      m_index[  range_rank  ] = domain_rank ;

      return set( domain_rank + 1 , range_rank + 1 , dim , args... )
#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )
             && ( val.size() == 2 )
             && ( e <= b + dim.extent( domain_rank ) )
#endif
      ;
    }

  //------------------------------

#if defined( KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK )

  template< size_t ... DimArgs >
  void error( char *
            , int
            , unsigned
            , unsigned
            , const ViewDimension< DimArgs ... > & ) const
    {}

  template< class T , size_t ... DimArgs , class ... Args >
  void error( char * buf , int buf_len
            , unsigned domain_rank
            , unsigned range_rank
            , const ViewDimension< DimArgs ... > & dim
            , const T & val
            , Args ... args ) const
    {
      const int n = std::min( buf_len ,
        snprintf( buf , buf_len
                , " %lu < %lu %c"
                , static_cast<unsigned long>(val)
                , static_cast<unsigned long>( dim.extent( domain_rank ) )
                , int( sizeof...(Args) ? ',' : ')' ) ) );

      error( buf+n, buf_len-n, domain_rank + 1 , range_rank , dim , args... );
    }

  // std::pair range
  template< size_t ... DimArgs , class ... Args >
  void error( char * buf , int buf_len
            , unsigned domain_rank
            , unsigned range_rank
            , const ViewDimension< DimArgs ... > & dim
            , const Kokkos::Impl::ALL_t
            , Args ... args ) const
    {
      const int n = std::min( buf_len ,
        snprintf( buf , buf_len
                , " Kokkos::ALL %c"
                , int( sizeof...(Args) ? ',' : ')' ) ) );

      error( buf+n , buf_len-n , domain_rank + 1 , range_rank + 1 , dim , args... );
    }

  // std::pair range
  template< class T , size_t ... DimArgs , class ... Args >
  void error( char * buf , int buf_len
            , unsigned domain_rank
            , unsigned range_rank
            , const ViewDimension< DimArgs ... > & dim
            , const std::pair<T,T> & val
            , Args ... args ) const
    {
      // d <= e - b
      const int n = std::min( buf_len ,
        snprintf( buf , buf_len
                , " %lu <= %lu - %lu %c"
                , static_cast<unsigned long>( dim.extent( domain_rank ) )
                , static_cast<unsigned long>( val.second )
                , static_cast<unsigned long>( val.first )
                , int( sizeof...(Args) ? ',' : ')' ) ) );

      error( buf+n , buf_len-n , domain_rank + 1 , range_rank + 1 , dim , args... );
    }

  // Kokkos::pair range
  template< class T , size_t ... DimArgs , class ... Args >
  void error( char * buf , int buf_len
            , unsigned domain_rank
            , unsigned range_rank
            , const ViewDimension< DimArgs ... > & dim
            , const Kokkos::pair<T,T> & val
            , Args ... args ) const
    {
      // d <= e - b
      const int n = std::min( buf_len ,
        snprintf( buf , buf_len
                , " %lu <= %lu - %lu %c"
                , static_cast<unsigned long>( dim.extent( domain_rank ) )
                , static_cast<unsigned long>( val.second )
                , static_cast<unsigned long>( val.first )
                , int( sizeof...(Args) ? ',' : ')' ) ) );

      error( buf+n , buf_len-n , domain_rank + 1 , range_rank + 1 , dim , args... );
    }

  // { begin , end } range
  template< class T , size_t ... DimArgs , class ... Args >
  void error( char * buf , int buf_len
            , unsigned domain_rank
            , unsigned range_rank
            , const ViewDimension< DimArgs ... > & dim
            , const std::initializer_list< T > & val
            , Args ... args ) const
    {
      // d <= e - b
      int n = 0 ;
      if ( val.size() == 2 ) {
        n = std::min( buf_len ,
          snprintf( buf , buf_len
                  , " %lu <= %lu - %lu %c"
                  , static_cast<unsigned long>( dim.extent( domain_rank ) )
                  , static_cast<unsigned long>( val.begin()[0] )
                  , static_cast<unsigned long>( val.begin()[1] )
                  , int( sizeof...(Args) ? ',' : ')' ) ) );
      }
      else {
        n = std::min( buf_len ,
          snprintf( buf , buf_len
                  , " { ... }.size() == %u %c"
                  , unsigned(val.size())
                  , int( sizeof...(Args) ? ',' : ')' ) ) );
      }

      error( buf+n , buf_len-n , domain_rank + 1 , range_rank + 1 , dim , args... );
    }

  template< size_t ... DimArgs , class ... Args >
  KOKKOS_FORCEINLINE_FUNCTION
  void error( const ViewDimension< DimArgs ... > & dim , Args ... args ) const
    {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
      enum { LEN = 1024 };
      char buffer[ LEN ];

      const int n = snprintf(buffer,LEN,"Kokkos::subview bounds error (");
      error( buffer+n , LEN-n , 0 , 0 , dim , args... );

      Kokkos::Impl::throw_runtime_exception(std::string(buffer));
#else
      Kokkos::abort("Kokkos::subview bounds error");
#endif
    }

#else

  template< size_t ... DimArgs , class ... Args >
  KOKKOS_FORCEINLINE_FUNCTION
  void error( const ViewDimension< DimArgs ... > & , Args ... ) const {}

#endif

public:

  template< size_t ... DimArgs , class ... Args >
  KOKKOS_INLINE_FUNCTION
  SubviewExtents( const ViewDimension< DimArgs ... > & dim , Args ... args )
    {
      static_assert( DomainRank == sizeof...(DimArgs) , "" );
      static_assert( DomainRank == sizeof...(Args) , "" );

      // Verifies that all arguments, up to 8, are integral types,
      // integral extents, or don't exist.
      static_assert( RangeRank ==
        unsigned( is_integral_extent<0,Args...>::value ) +
        unsigned( is_integral_extent<1,Args...>::value ) +
        unsigned( is_integral_extent<2,Args...>::value ) +
        unsigned( is_integral_extent<3,Args...>::value ) +
        unsigned( is_integral_extent<4,Args...>::value ) +
        unsigned( is_integral_extent<5,Args...>::value ) +
        unsigned( is_integral_extent<6,Args...>::value ) +
        unsigned( is_integral_extent<7,Args...>::value ) , "" );

      if ( RangeRank == 0 ) { m_length[0] = 0 ; m_index[0] = ~0u ; }

      if ( ! set( 0 , 0 , dim , args... ) ) error( dim , args... );
    }

  template < typename iType >
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr size_t domain_offset( const iType i ) const
    { return unsigned(i) < DomainRank ? m_begin[i] : 0 ; }

  template < typename iType >
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr size_t range_extent( const iType i ) const
    { return unsigned(i) < InternalRangeRank ? m_length[i] : 0 ; }

  template < typename iType >
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr unsigned range_index( const iType i ) const
    { return unsigned(i) < InternalRangeRank ? m_index[i] : ~0u ; }
};

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  Given a value type and dimension generate the View data type */
template< class T , class Dim >
struct ViewDataType ;

template< class T >
struct ViewDataType< T , ViewDimension<> >
{
  typedef T type ;
};

template< class T , size_t ... Args >
struct ViewDataType< T , ViewDimension< 0 , Args... > >
{
  typedef typename ViewDataType<T*,ViewDimension<Args...> >::type type ;
};

template< class T , size_t N , size_t ... Args >
struct ViewDataType< T , ViewDimension< N , Args... > >
{
  typedef typename ViewDataType<T,ViewDimension<Args...> >::type type[N] ;
};

/**\brief  Analysis of View data type.
 *
 *  Data type conforms to one of the following patterns :
 *    {const} value_type [][#][#][#]
 *    {const} value_type ***[#][#][#]
 *  Where the sum of counts of '*' and '[#]' is at most ten.
 *
 *  Provide typedef for the ViewDimension<...> and value_type.
 */
template< class T >
struct ViewArrayAnalysis
{
  typedef T                                      value_type ;
  typedef typename std::add_const<    T >::type  const_value_type ;
  typedef typename std::remove_const< T >::type  non_const_value_type ;
  typedef ViewDimension<>                        static_dimension ;
  typedef ViewDimension<>                        dynamic_dimension ;
  typedef ViewDimension<>                        dimension ;
};

template< class T , size_t N >
struct ViewArrayAnalysis< T[N] >
{
private:
  typedef ViewArrayAnalysis< T > nested ;
public:
  typedef typename nested::value_type            value_type ;
  typedef typename nested::const_value_type      const_value_type ;
  typedef typename nested::non_const_value_type  non_const_value_type ;

  typedef typename nested::static_dimension::template prepend<N>::type
    static_dimension ;

  typedef typename nested::dynamic_dimension dynamic_dimension ;

  typedef typename
    ViewDimensionJoin< dynamic_dimension , static_dimension >::type
      dimension ;
};

template< class T >
struct ViewArrayAnalysis< T[] >
{
private:
  typedef ViewArrayAnalysis< T > nested ;
  typedef typename nested::dimension nested_dimension ;
public:
  typedef typename nested::value_type            value_type ;
  typedef typename nested::const_value_type      const_value_type ;
  typedef typename nested::non_const_value_type  non_const_value_type ;

  typedef typename nested::dynamic_dimension::template prepend<0>::type
    dynamic_dimension ;

  typedef typename nested::static_dimension static_dimension ;

  typedef typename
    ViewDimensionJoin< dynamic_dimension , static_dimension >::type
      dimension ;
};

template< class T >
struct ViewArrayAnalysis< T* >
{
private:
  typedef ViewArrayAnalysis< T > nested ;
public:
  typedef typename nested::value_type            value_type ;
  typedef typename nested::const_value_type      const_value_type ;
  typedef typename nested::non_const_value_type  non_const_value_type ;

  typedef typename nested::dynamic_dimension::template prepend<0>::type
    dynamic_dimension ;

  typedef typename nested::static_dimension static_dimension ;

  typedef typename
    ViewDimensionJoin< dynamic_dimension , static_dimension >::type
      dimension ;
};


template< class DataType , class ArrayLayout , class ValueType >
struct ViewDataAnalysis
{
private:

  typedef ViewArrayAnalysis< DataType > array_analysis ;

  // ValueType is opportunity for partial specialization.
  // Must match array analysis when this default template is used.
  static_assert( std::is_same< ValueType , typename array_analysis::non_const_value_type >::value , "" );

public:

  typedef void specialize ; // No specialization

  typedef typename array_analysis::dimension             dimension ;
  typedef typename array_analysis::value_type            value_type ;
  typedef typename array_analysis::const_value_type      const_value_type ;
  typedef typename array_analysis::non_const_value_type  non_const_value_type ;

  // Generate analogous multidimensional array specification type.
  typedef typename ViewDataType<           value_type , dimension >::type  type ;
  typedef typename ViewDataType<     const_value_type , dimension >::type  const_type ;
  typedef typename ViewDataType< non_const_value_type , dimension >::type  non_const_type ;

  // Generate "flattened" multidimensional array specification type.
  typedef type            scalar_array_type ;
  typedef const_type      const_scalar_array_type ;
  typedef non_const_type  non_const_scalar_array_type ;
};

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template < class Dimension , class Layout , typename Enable = void >
struct ViewOffset {
  using is_mapping_plugin = std::false_type ;
};

//----------------------------------------------------------------------------
// LayoutLeft AND ( 1 >= rank OR 0 == rank_dynamic ) : no padding / striding
template < class Dimension >
struct ViewOffset< Dimension , Kokkos::LayoutLeft
                 , typename std::enable_if<( 1 >= Dimension::rank
                                             ||
                                             0 == Dimension::rank_dynamic
                                           )>::type >
{
  using is_mapping_plugin = std::true_type ;
  using is_regular        = std::true_type ;

  typedef size_t             size_type ;
  typedef Dimension          dimension_type ;
  typedef Kokkos::LayoutLeft array_layout ;

  dimension_type m_dim ;

  //----------------------------------------

  // rank 1
  template< typename I0 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 ) const { return i0 ; }

  // rank 2
  template < typename I0 , typename I1 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 , I1 const & i1 ) const
    { return i0 + m_dim.N0 * i1 ; }

  //rank 3
  template < typename I0, typename I1, typename I2 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2 ) const
  {
    return i0 + m_dim.N0 * ( i1 + m_dim.N1 * i2 );
  }

  //rank 4
  template < typename I0, typename I1, typename I2, typename I3 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3 ) const
  {
    return i0 + m_dim.N0 * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * i3 ));
  }

  //rank 5
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4 ) const
  {
    return i0 + m_dim.N0 * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * i4 )));
  }

  //rank 6
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5 ) const
  {
    return i0 + m_dim.N0 * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * (
           i4 + m_dim.N4 * i5 ))));
  }

  //rank 7
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6 ) const
  {
    return i0 + m_dim.N0 * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * (
           i4 + m_dim.N4 * (
           i5 + m_dim.N5 * i6 )))));
  }

  //rank 8
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6, typename I7 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6, I7 const & i7 ) const
  {
    return i0 + m_dim.N0 * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * (
           i4 + m_dim.N4 * (
           i5 + m_dim.N5 * (
           i6 + m_dim.N6 * i7 ))))));
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const
    {
      return array_layout( m_dim.N0 , m_dim.N1 , m_dim.N2 , m_dim.N3
                         , m_dim.N4 , m_dim.N5 , m_dim.N6 , m_dim.N7 );
    }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const { return m_dim.N0 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const { return m_dim.N1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const { return m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const { return m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const { return m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const { return m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const { return m_dim.N6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const { return m_dim.N7 ; }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const
    { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const
    { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const { return true ; }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return 1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const { return m_dim.N0 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const { return m_dim.N0 * m_dim.N1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const { return m_dim.N0 * m_dim.N1 * m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 ; }

  // Stride with [ rank ] value is the total length
  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  void stride( iType * const s ) const
    {
      s[0] = 1 ;
      if ( 0 < dimension_type::rank ) { s[1] = m_dim.N0 ; }
      if ( 1 < dimension_type::rank ) { s[2] = s[1] * m_dim.N1 ; }
      if ( 2 < dimension_type::rank ) { s[3] = s[2] * m_dim.N2 ; }
      if ( 3 < dimension_type::rank ) { s[4] = s[3] * m_dim.N3 ; }
      if ( 4 < dimension_type::rank ) { s[5] = s[4] * m_dim.N4 ; }
      if ( 5 < dimension_type::rank ) { s[6] = s[5] * m_dim.N5 ; }
      if ( 6 < dimension_type::rank ) { s[7] = s[6] * m_dim.N6 ; }
      if ( 7 < dimension_type::rank ) { s[8] = s[7] * m_dim.N7 ; }
    }

  //----------------------------------------

  ViewOffset() = default ;
  ViewOffset( const ViewOffset & ) = default ;
  ViewOffset & operator = ( const ViewOffset & ) = default ;

  template< unsigned TrivialScalarSize >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( std::integral_constant<unsigned,TrivialScalarSize> const &
    , Kokkos::LayoutLeft const & arg_layout
    )
    : m_dim( arg_layout.dimension[0], 0, 0, 0, 0, 0, 0, 0 )
    {}

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutLeft , void > & rhs )
    : m_dim( rhs.m_dim.N0 , rhs.m_dim.N1 , rhs.m_dim.N2 , rhs.m_dim.N3
           , rhs.m_dim.N4 , rhs.m_dim.N5 , rhs.m_dim.N6 , rhs.m_dim.N7 )
    {
      static_assert( int(DimRHS::rank) == int(dimension_type::rank) , "ViewOffset assignment requires equal rank" );
      // Also requires equal static dimensions ...
    }

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutRight , void > & rhs )
    : m_dim( rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0 )
    {
      static_assert(
        ( DimRHS::rank == 0 &&
          dimension_type::rank == 0 ) ||
        ( DimRHS::rank == 1 &&
          dimension_type::rank == 1 &&
          dimension_type::rank_dynamic == 1 )
        , "ViewOffset LayoutLeft and LayoutRight are only compatible when rank <= 1" );
    }

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutStride , void > & rhs )
    : m_dim( rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0 )
    {
      static_assert( 
        ( DimRHS::rank == 0 &&
          dimension_type::rank == 0 ) ||
        ( DimRHS::rank == 1 &&
          dimension_type::rank == 1 &&
          dimension_type::rank_dynamic == 1 )
        , "ViewOffset LayoutLeft and LayoutStride are only compatible when rank <= 1" );
      if ( rhs.m_stride.S0 != 1 ) {
        Kokkos::abort("Kokkos::Impl::ViewOffset assignment of LayoutLeft from LayoutStride  requires stride == 1" );
      }
    }

  //----------------------------------------
  // Subview construction

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset(
    const ViewOffset< DimRHS , Kokkos::LayoutLeft , void > & ,
    const SubviewExtents< DimRHS::rank , dimension_type::rank > & sub )
    : m_dim( sub.range_extent(0), 0, 0, 0, 0, 0, 0, 0 )
    {
      static_assert( ( 0 == dimension_type::rank ) ||
                     ( 1 == dimension_type::rank && 1 == dimension_type::rank_dynamic && 1 <= DimRHS::rank )
                   , "ViewOffset subview construction requires compatible rank" );
    }
};

//----------------------------------------------------------------------------
// LayoutLeft AND ( 1 < rank AND 0 < rank_dynamic ) : has padding / striding
template < class Dimension >
struct ViewOffset< Dimension , Kokkos::LayoutLeft
                 , typename std::enable_if<( 1 < Dimension::rank
                                             &&
                                             0 < Dimension::rank_dynamic
                                           )>::type >
{
  using is_mapping_plugin = std::true_type ;
  using is_regular        = std::true_type ;

  typedef size_t             size_type ;
  typedef Dimension          dimension_type ;
  typedef Kokkos::LayoutLeft array_layout ;

  dimension_type m_dim ;
  size_type      m_stride ;

  //----------------------------------------

  // rank 1
  template< typename I0 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 ) const { return i0 ; }

  // rank 2
  template < typename I0 , typename I1 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 , I1 const & i1 ) const
    { return i0 + m_stride * i1 ; }

  //rank 3
  template < typename I0, typename I1, typename I2 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2 ) const
  {
    return i0 + m_stride * ( i1 + m_dim.N1 * i2 );
  }

  //rank 4
  template < typename I0, typename I1, typename I2, typename I3 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3 ) const
  {
    return i0 + m_stride * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * i3 ));
  }

  //rank 5
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4 ) const
  {
    return i0 + m_stride * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * i4 )));
  }

  //rank 6
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5 ) const
  {
    return i0 + m_stride * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * (
           i4 + m_dim.N4 * i5 ))));
  }

  //rank 7
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6 ) const
  {
    return i0 + m_stride * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * (
           i4 + m_dim.N4 * (
           i5 + m_dim.N5 * i6 )))));
  }

  //rank 8
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6, typename I7 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6, I7 const & i7 ) const
  {
    return i0 + m_stride * (
           i1 + m_dim.N1 * (
           i2 + m_dim.N2 * (
           i3 + m_dim.N3 * (
           i4 + m_dim.N4 * (
           i5 + m_dim.N5 * (
           i6 + m_dim.N6 * i7 ))))));
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const
    {
      return array_layout( m_dim.N0 , m_dim.N1 , m_dim.N2 , m_dim.N3
                         , m_dim.N4 , m_dim.N5 , m_dim.N6 , m_dim.N7 );
    }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const { return m_dim.N0 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const { return m_dim.N1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const { return m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const { return m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const { return m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const { return m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const { return m_dim.N6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const { return m_dim.N7 ; }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const
    { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const
    { return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const { return m_stride == m_dim.N0 ; }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return 1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const { return m_stride ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const { return m_stride * m_dim.N1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const { return m_stride * m_dim.N1 * m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const { return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const { return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const { return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 ; }

  // Stride with [ rank ] value is the total length
  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  void stride( iType * const s ) const
    {
      s[0] = 1 ;
      if ( 0 < dimension_type::rank ) { s[1] = m_stride ; }
      if ( 1 < dimension_type::rank ) { s[2] = s[1] * m_dim.N1 ; }
      if ( 2 < dimension_type::rank ) { s[3] = s[2] * m_dim.N2 ; }
      if ( 3 < dimension_type::rank ) { s[4] = s[3] * m_dim.N3 ; }
      if ( 4 < dimension_type::rank ) { s[5] = s[4] * m_dim.N4 ; }
      if ( 5 < dimension_type::rank ) { s[6] = s[5] * m_dim.N5 ; }
      if ( 6 < dimension_type::rank ) { s[7] = s[6] * m_dim.N6 ; }
      if ( 7 < dimension_type::rank ) { s[8] = s[7] * m_dim.N7 ; }
    }

  //----------------------------------------

private:

  template< unsigned TrivialScalarSize >
  struct Padding {
    enum { div = TrivialScalarSize == 0 ? 0 : Kokkos::Impl::MEMORY_ALIGNMENT / ( TrivialScalarSize ? TrivialScalarSize : 1 ) };
    enum { mod = TrivialScalarSize == 0 ? 0 : Kokkos::Impl::MEMORY_ALIGNMENT % ( TrivialScalarSize ? TrivialScalarSize : 1 ) };

    // If memory alignment is a multiple of the trivial scalar size then attempt to align.
    enum { align = 0 != TrivialScalarSize && 0 == mod ? div : 0 };
    enum { div_ok = (div != 0) ? div : 1 }; // To valid modulo zero in constexpr

    KOKKOS_INLINE_FUNCTION
    static constexpr size_t stride( size_t const N )
    {
      return ( (align != 0) && ((Kokkos::Impl::MEMORY_ALIGNMENT_THRESHOLD * align) < N) && ((N % div_ok) != 0) )
             ? N + align - ( N % div_ok ) : N ;
    }
  };

public:

  ViewOffset() = default ;
  ViewOffset( const ViewOffset & ) = default ;
  ViewOffset & operator = ( const ViewOffset & ) = default ;

  /* Enable padding for trivial scalar types with non-zero trivial scalar size */
  template< unsigned TrivialScalarSize >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( std::integral_constant<unsigned,TrivialScalarSize> const &
    , Kokkos::LayoutLeft const & arg_layout
    )
    : m_dim( arg_layout.dimension[0] , arg_layout.dimension[1]
           , arg_layout.dimension[2] , arg_layout.dimension[3]
           , arg_layout.dimension[4] , arg_layout.dimension[5]
           , arg_layout.dimension[6] , arg_layout.dimension[7]
           )
    , m_stride( Padding<TrivialScalarSize>::stride( arg_layout.dimension[0] ) )
    {}

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutLeft , void > & rhs )
    : m_dim( rhs.m_dim.N0 , rhs.m_dim.N1 , rhs.m_dim.N2 , rhs.m_dim.N3
           , rhs.m_dim.N4 , rhs.m_dim.N5 , rhs.m_dim.N6 , rhs.m_dim.N7 )
    , m_stride( rhs.stride_1() )
    {
      static_assert( int(DimRHS::rank) == int(dimension_type::rank) , "ViewOffset assignment requires equal rank" );
      // Also requires equal static dimensions ...
    }

  //----------------------------------------
  // Subview construction
  // This subview must be 2 == rank and 2 == rank_dynamic
  // due to only having stride #0.
  // The source dimension #0 must be non-zero for stride-one leading dimension.
  // At most subsequent dimension can be non-zero.

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( const ViewOffset< DimRHS , Kokkos::LayoutLeft , void > & rhs ,
      const SubviewExtents< DimRHS::rank , dimension_type::rank > & sub )
    : m_dim( sub.range_extent(0)
           , sub.range_extent(1)
           , sub.range_extent(2)
           , sub.range_extent(3)
           , sub.range_extent(4)
           , sub.range_extent(5)
           , sub.range_extent(6)
           , sub.range_extent(7))
    , m_stride( ( 1 == sub.range_index(1) ? rhs.stride_1() :
                ( 2 == sub.range_index(1) ? rhs.stride_2() :
                ( 3 == sub.range_index(1) ? rhs.stride_3() :
                ( 4 == sub.range_index(1) ? rhs.stride_4() :
                ( 5 == sub.range_index(1) ? rhs.stride_5() :
                ( 6 == sub.range_index(1) ? rhs.stride_6() :
                ( 7 == sub.range_index(1) ? rhs.stride_7() : 0 ))))))))
    {
      //static_assert( ( 2 == dimension_type::rank ) &&
      //               ( 2 == dimension_type::rank_dynamic ) &&
      //               ( 2 <= DimRHS::rank )
      //             , "ViewOffset subview construction requires compatible rank" );
    }
};

//----------------------------------------------------------------------------
// LayoutRight AND ( 1 >= rank OR 0 == rank_dynamic ) : no padding / striding
template < class Dimension >
struct ViewOffset< Dimension , Kokkos::LayoutRight
                 , typename std::enable_if<( 1 >= Dimension::rank
                                             ||
                                             0 == Dimension::rank_dynamic
                                           )>::type >
{
  using is_mapping_plugin = std::true_type ;
  using is_regular        = std::true_type ;

  typedef size_t              size_type ;
  typedef Dimension           dimension_type ;
  typedef Kokkos::LayoutRight array_layout ;

  dimension_type m_dim ;

  //----------------------------------------

  // rank 1
  template< typename I0 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 ) const { return i0 ; }

  // rank 2
  template < typename I0 , typename I1 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 , I1 const & i1 ) const
    { return i1 + m_dim.N1 * i0 ; }

  //rank 3
  template < typename I0, typename I1, typename I2 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2 ) const
  {
    return i2 + m_dim.N2 * ( i1 + m_dim.N1 * ( i0 ));
  }

  //rank 4
  template < typename I0, typename I1, typename I2, typename I3 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3 ) const
  {
    return i3 + m_dim.N3 * (
           i2 + m_dim.N2 * (
           i1 + m_dim.N1 * ( i0 )));
  }

  //rank 5
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4 ) const
  {
    return i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * (
           i1 + m_dim.N1 * ( i0 ))));
  }

  //rank 6
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5 ) const
  {
    return i5 + m_dim.N5 * (
           i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * (
           i1 + m_dim.N1 * ( i0 )))));
  }

  //rank 7
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6 ) const
  {
    return i6 + m_dim.N6 * (
           i5 + m_dim.N5 * (
           i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * (
           i1 + m_dim.N1 * ( i0 ))))));
  }

  //rank 8
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6, typename I7 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6, I7 const & i7 ) const
  {
    return i7 + m_dim.N7 * (
           i6 + m_dim.N6 * (
           i5 + m_dim.N5 * (
           i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * (
           i1 + m_dim.N1 * ( i0 )))))));
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const
    {
      return array_layout( m_dim.N0 , m_dim.N1 , m_dim.N2 , m_dim.N3
                         , m_dim.N4 , m_dim.N5 , m_dim.N6 , m_dim.N7 );
    }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const { return m_dim.N0 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const { return m_dim.N1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const { return m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const { return m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const { return m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const { return m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const { return m_dim.N6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const { return m_dim.N7 ; }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const
    { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const
    { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const { return true ; }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return 1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const { return m_dim.N7 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const { return m_dim.N7 * m_dim.N6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2 * m_dim.N1 ; }

  // Stride with [ rank ] value is the total length
  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  void stride( iType * const s ) const
    {
      size_type n = 1 ;
      if ( 7 < dimension_type::rank ) { s[7] = n ; n *= m_dim.N7 ; }
      if ( 6 < dimension_type::rank ) { s[6] = n ; n *= m_dim.N6 ; }
      if ( 5 < dimension_type::rank ) { s[5] = n ; n *= m_dim.N5 ; }
      if ( 4 < dimension_type::rank ) { s[4] = n ; n *= m_dim.N4 ; }
      if ( 3 < dimension_type::rank ) { s[3] = n ; n *= m_dim.N3 ; }
      if ( 2 < dimension_type::rank ) { s[2] = n ; n *= m_dim.N2 ; }
      if ( 1 < dimension_type::rank ) { s[1] = n ; n *= m_dim.N1 ; }
      if ( 0 < dimension_type::rank ) { s[0] = n ; }
      s[dimension_type::rank] = n * m_dim.N0 ;
    }

  //----------------------------------------

  ViewOffset() = default ;
  ViewOffset( const ViewOffset & ) = default ;
  ViewOffset & operator = ( const ViewOffset & ) = default ;

  template< unsigned TrivialScalarSize >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( std::integral_constant<unsigned,TrivialScalarSize> const &
    , Kokkos::LayoutRight const & arg_layout
    )
    : m_dim( arg_layout.dimension[0], 0, 0, 0, 0, 0, 0, 0 )
    {}

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutRight , void > & rhs )
    : m_dim( rhs.m_dim.N0 , rhs.m_dim.N1 , rhs.m_dim.N2 , rhs.m_dim.N3
           , rhs.m_dim.N4 , rhs.m_dim.N5 , rhs.m_dim.N6 , rhs.m_dim.N7 )
    {
      static_assert( int(DimRHS::rank) == int(dimension_type::rank) , "ViewOffset assignment requires equal rank" );
      // Also requires equal static dimensions ...
    }

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutLeft , void > & rhs )
    : m_dim( rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0 )
    {
      static_assert(
       ( DimRHS::rank == 0 &&
         dimension_type::rank == 0 ) ||
       ( DimRHS::rank == 1 &&
         dimension_type::rank == 1 &&
         dimension_type::rank_dynamic == 1 )
      , "ViewOffset LayoutRight and LayoutLeft are only compatible when rank <= 1" );
    }

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutStride , void > & rhs )
    : m_dim( rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0 )
    {
      static_assert(
       ( DimRHS::rank == 0 &&
         dimension_type::rank == 0 ) ||
       ( DimRHS::rank == 1 &&
         dimension_type::rank == 1 &&
         dimension_type::rank_dynamic == 1 )
      , "ViewOffset LayoutRight and LayoutString are only compatible when rank <= 1" );
      if ( rhs.m_stride.S0 != 1 ) {
        Kokkos::abort("Kokkos::Impl::ViewOffset assignment of LayoutLeft/Right from LayoutStride  requires stride == 1" );
      }
    }

  //----------------------------------------
  // Subview construction

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( const ViewOffset< DimRHS , Kokkos::LayoutRight , void > & rhs
    , const SubviewExtents< DimRHS::rank , dimension_type::rank > & sub
    )
    : m_dim( sub.range_extent(0) , 0, 0, 0, 0, 0, 0, 0 )
    {
      static_assert( ( 0 == dimension_type::rank_dynamic ) ||
                     ( 1 == dimension_type::rank && 1 == dimension_type::rank_dynamic && 1 <= DimRHS::rank )
                   , "ViewOffset subview construction requires compatible rank" );
    }
};

//----------------------------------------------------------------------------
// LayoutRight AND ( 1 < rank AND 0 < rank_dynamic ) : has padding / striding
template < class Dimension >
struct ViewOffset< Dimension , Kokkos::LayoutRight
                 , typename std::enable_if<( 1 < Dimension::rank
                                             &&
                                             0 < Dimension::rank_dynamic
                                           )>::type >
{
  using is_mapping_plugin = std::true_type ;
  using is_regular        = std::true_type ;

  typedef size_t               size_type ;
  typedef Dimension            dimension_type ;
  typedef Kokkos::LayoutRight  array_layout ;

  dimension_type m_dim ;
  size_type      m_stride ;

  //----------------------------------------

  // rank 1
  template< typename I0 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 ) const { return i0 ; }

  // rank 2
  template < typename I0 , typename I1 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 , I1 const & i1 ) const
  { return i1 + i0 * m_stride ; }

  //rank 3
  template < typename I0, typename I1, typename I2 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2 ) const
  { return i2 + m_dim.N2 * ( i1 ) + i0 * m_stride ; }

  //rank 4
  template < typename I0, typename I1, typename I2, typename I3 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3 ) const
  {
    return i3 + m_dim.N3 * (
           i2 + m_dim.N2 * ( i1 )) +
           i0 * m_stride ;
  }

  //rank 5
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4 ) const
  {
    return i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * ( i1 ))) +
           i0 * m_stride ;
  }

  //rank 6
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5 ) const
  {
    return i5 + m_dim.N5 * (
           i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * ( i1 )))) +
           i0 * m_stride ;
  }

  //rank 7
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6 ) const
  {
    return i6 + m_dim.N6 * (
           i5 + m_dim.N5 * (
           i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * ( i1 ))))) +
           i0 * m_stride ;
  }

  //rank 8
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6, typename I7 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6, I7 const & i7 ) const
  {
    return i7 + m_dim.N7 * (
           i6 + m_dim.N6 * (
           i5 + m_dim.N5 * (
           i4 + m_dim.N4 * (
           i3 + m_dim.N3 * (
           i2 + m_dim.N2 * ( i1 )))))) +
           i0 * m_stride ;
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const
    {
      return array_layout( m_dim.N0 , m_dim.N1 , m_dim.N2 , m_dim.N3
                         , m_dim.N4 , m_dim.N5 , m_dim.N6 , m_dim.N7 );
    }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const { return m_dim.N0 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const { return m_dim.N1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const { return m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const { return m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const { return m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const { return m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const { return m_dim.N6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const { return m_dim.N7 ; }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const
    { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

  /* Span of the range space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const
    { return m_dim.N0 * m_stride ; }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const
    { return m_stride == m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2 * m_dim.N1 ; }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return 1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const { return m_dim.N7 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const { return m_dim.N7 * m_dim.N6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const { return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return m_stride ; }

  // Stride with [ rank ] value is the total length
  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  void stride( iType * const s ) const
    {
      size_type n = 1 ;
      if ( 7 < dimension_type::rank ) { s[7] = n ; n *= m_dim.N7 ; }
      if ( 6 < dimension_type::rank ) { s[6] = n ; n *= m_dim.N6 ; }
      if ( 5 < dimension_type::rank ) { s[5] = n ; n *= m_dim.N5 ; }
      if ( 4 < dimension_type::rank ) { s[4] = n ; n *= m_dim.N4 ; }
      if ( 3 < dimension_type::rank ) { s[3] = n ; n *= m_dim.N3 ; }
      if ( 2 < dimension_type::rank ) { s[2] = n ; n *= m_dim.N2 ; }
      if ( 1 < dimension_type::rank ) { s[1] = n ; }
      if ( 0 < dimension_type::rank ) { s[0] = m_stride ; }
      s[dimension_type::rank] = m_stride * m_dim.N0 ;
    }

  //----------------------------------------

private:

  template< unsigned TrivialScalarSize >
  struct Padding {
    enum { div = TrivialScalarSize == 0 ? 0 : Kokkos::Impl::MEMORY_ALIGNMENT / ( TrivialScalarSize ? TrivialScalarSize : 1 ) };
    enum { mod = TrivialScalarSize == 0 ? 0 : Kokkos::Impl::MEMORY_ALIGNMENT % ( TrivialScalarSize ? TrivialScalarSize : 1 ) };

    // If memory alignment is a multiple of the trivial scalar size then attempt to align.
    enum { align = 0 != TrivialScalarSize && 0 == mod ? div : 0 };
    enum { div_ok = (div != 0) ? div : 1 }; // To valid modulo zero in constexpr

    KOKKOS_INLINE_FUNCTION
    static constexpr size_t stride( size_t const N )
    {
      return ( (align != 0) && ((Kokkos::Impl::MEMORY_ALIGNMENT_THRESHOLD * align) < N) && ((N % div_ok) != 0) )
             ? N + align - ( N % div_ok ) : N ;
    }
  };

public:

  ViewOffset() = default ;
  ViewOffset( const ViewOffset & ) = default ;
  ViewOffset & operator = ( const ViewOffset & ) = default ;

  /* Enable padding for trivial scalar types with non-zero trivial scalar size.  */
  template< unsigned TrivialScalarSize >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( std::integral_constant<unsigned,TrivialScalarSize> const &
    , Kokkos::LayoutRight const & arg_layout
    )
    : m_dim( arg_layout.dimension[0] , arg_layout.dimension[1]
           , arg_layout.dimension[2] , arg_layout.dimension[3]
           , arg_layout.dimension[4] , arg_layout.dimension[5]
           , arg_layout.dimension[6] , arg_layout.dimension[7]
           )
    , m_stride( Padding<TrivialScalarSize>::
                  stride( /* 2 <= rank */
                          m_dim.N1 * ( dimension_type::rank == 2 ? 1 :
                          m_dim.N2 * ( dimension_type::rank == 3 ? 1 :
                          m_dim.N3 * ( dimension_type::rank == 4 ? 1 :
                          m_dim.N4 * ( dimension_type::rank == 5 ? 1 :
                          m_dim.N5 * ( dimension_type::rank == 6 ? 1 :
                          m_dim.N6 * ( dimension_type::rank == 7 ? 1 : m_dim.N7 )))))) ))
    {}

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( const ViewOffset< DimRHS , Kokkos::LayoutRight , void > & rhs )
    : m_dim( rhs.m_dim.N0 , rhs.m_dim.N1 , rhs.m_dim.N2 , rhs.m_dim.N3
           , rhs.m_dim.N4 , rhs.m_dim.N5 , rhs.m_dim.N6 , rhs.m_dim.N7 )
    , m_stride( rhs.stride_0() )
    {
      static_assert( int(DimRHS::rank) == int(dimension_type::rank) , "ViewOffset assignment requires equal rank" );
      // Also requires equal static dimensions ...
    }

  //----------------------------------------
  // Subview construction
  // Last dimension must be non-zero

  template< class DimRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( const ViewOffset< DimRHS , Kokkos::LayoutRight , void > & rhs
    , const SubviewExtents< DimRHS::rank , dimension_type::rank > & sub
    )
    : m_dim( sub.range_extent(0)
           , sub.range_extent(1)
           , sub.range_extent(2)
           , sub.range_extent(3)
           , sub.range_extent(4)
           , sub.range_extent(5)
           , sub.range_extent(6)
           , sub.range_extent(7))
    , m_stride( 0 == sub.range_index(0) ? rhs.stride_0() : (
                1 == sub.range_index(0) ? rhs.stride_1() : (
                2 == sub.range_index(0) ? rhs.stride_2() : (
                3 == sub.range_index(0) ? rhs.stride_3() : (
                4 == sub.range_index(0) ? rhs.stride_4() : (
                5 == sub.range_index(0) ? rhs.stride_5() : (
                6 == sub.range_index(0) ? rhs.stride_6() : 0 )))))))
    {
/*      // This subview must be 2 == rank and 2 == rank_dynamic
      // due to only having stride #0.
      // The source dimension #0 must be non-zero for stride-one leading dimension.
      // At most subsequent dimension can be non-zero.

      static_assert( (( 2 == dimension_type::rank ) &&
                      ( 2 <= DimRHS::rank )) ||
                     ()
                   , "ViewOffset subview construction requires compatible rank" );
*/
    }
};

//----------------------------------------------------------------------------
/* Strided array layout only makes sense for 0 < rank */
/* rank = 0 included for DynRankView case */

template< unsigned Rank >
struct ViewStride ;

template<>
struct ViewStride<0> {
  enum { S0 = 0 , S1 = 0 , S2 = 0 , S3 = 0 , S4 = 0 , S5 = 0 , S6 = 0 , S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t , size_t , size_t , size_t
                      , size_t , size_t , size_t , size_t )
    {}
};

template<>
struct ViewStride<1> {
  size_t S0 ;
  enum { S1 = 0 , S2 = 0 , S3 = 0 , S4 = 0 , S5 = 0 , S6 = 0 , S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t , size_t , size_t
                      , size_t , size_t , size_t , size_t )
    : S0( aS0 )
    {}
};

template<>
struct ViewStride<2> {
  size_t S0 , S1 ;
  enum { S2 = 0 , S3 = 0 , S4 = 0 , S5 = 0 , S6 = 0 , S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t aS1 , size_t , size_t
                      , size_t , size_t , size_t , size_t )
    : S0( aS0 ) , S1( aS1 )
    {}
};

template<>
struct ViewStride<3> {
  size_t S0 , S1 , S2 ;
  enum { S3 = 0 , S4 = 0 , S5 = 0 , S6 = 0 , S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t aS1 , size_t aS2 , size_t
                      , size_t , size_t , size_t , size_t )
    : S0( aS0 ) , S1( aS1 ) , S2( aS2 )
    {}
};

template<>
struct ViewStride<4> {
  size_t S0 , S1 , S2 , S3 ;
  enum { S4 = 0 , S5 = 0 , S6 = 0 , S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t aS1 , size_t aS2 , size_t aS3
                      , size_t , size_t , size_t , size_t )
    : S0( aS0 ) , S1( aS1 ) , S2( aS2 ) , S3( aS3 )
    {}
};

template<>
struct ViewStride<5> {
  size_t S0 , S1 , S2 , S3 , S4 ;
  enum { S5 = 0 , S6 = 0 , S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t aS1 , size_t aS2 , size_t aS3
                      , size_t aS4 , size_t , size_t , size_t )
    : S0( aS0 ) , S1( aS1 ) , S2( aS2 ) , S3( aS3 )
    , S4( aS4 )
    {}
};

template<>
struct ViewStride<6> {
  size_t S0 , S1 , S2 , S3 , S4 , S5 ;
  enum { S6 = 0 , S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t aS1 , size_t aS2 , size_t aS3
                      , size_t aS4 , size_t aS5 , size_t , size_t )
    : S0( aS0 ) , S1( aS1 ) , S2( aS2 ) , S3( aS3 )
    , S4( aS4 ) , S5( aS5 )
    {}
};

template<>
struct ViewStride<7> {
  size_t S0 , S1 , S2 , S3 , S4 , S5 , S6 ;
  enum { S7 = 0 };

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t aS1 , size_t aS2 , size_t aS3
                      , size_t aS4 , size_t aS5 , size_t aS6 , size_t )
    : S0( aS0 ) , S1( aS1 ) , S2( aS2 ) , S3( aS3 )
    , S4( aS4 ) , S5( aS5 ) , S6( aS6 )
    {}
};

template<>
struct ViewStride<8> {
  size_t S0 , S1 , S2 , S3 , S4 , S5 , S6 , S7 ;

  ViewStride() = default ;
  ViewStride( const ViewStride & ) = default ;
  ViewStride & operator = ( const ViewStride & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewStride( size_t aS0 , size_t aS1 , size_t aS2 , size_t aS3
                      , size_t aS4 , size_t aS5 , size_t aS6 , size_t aS7 )
    : S0( aS0 ) , S1( aS1 ) , S2( aS2 ) , S3( aS3 )
    , S4( aS4 ) , S5( aS5 ) , S6( aS6 ) , S7( aS7 )
    {}
};

template < class Dimension >
struct ViewOffset< Dimension , Kokkos::LayoutStride
                 , void >
{
private:
  typedef ViewStride< Dimension::rank >  stride_type ;
public:

  using is_mapping_plugin = std::true_type ;
  using is_regular        = std::true_type ;

  typedef size_t                size_type ;
  typedef Dimension             dimension_type ;
  typedef Kokkos::LayoutStride  array_layout ;

  dimension_type  m_dim ;
  stride_type     m_stride ;

  //----------------------------------------

  // rank 1
  template< typename I0 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 ) const
  {
    return i0 * m_stride.S0 ;
  }

  // rank 2
  template < typename I0 , typename I1 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0 , I1 const & i1 ) const
  {
    return i0 * m_stride.S0 +
           i1 * m_stride.S1 ;
  }

  //rank 3
  template < typename I0, typename I1, typename I2 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2 ) const
  {
    return i0 * m_stride.S0 +
           i1 * m_stride.S1 +
           i2 * m_stride.S2 ;
  }

  //rank 4
  template < typename I0, typename I1, typename I2, typename I3 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3 ) const
  {
    return i0 * m_stride.S0 +
           i1 * m_stride.S1 +
           i2 * m_stride.S2 +
           i3 * m_stride.S3 ;
  }

  //rank 5
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4 ) const
  {
    return i0 * m_stride.S0 +
           i1 * m_stride.S1 +
           i2 * m_stride.S2 +
           i3 * m_stride.S3 +
           i4 * m_stride.S4 ;
  }

  //rank 6
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5 ) const
  {
    return i0 * m_stride.S0 +
           i1 * m_stride.S1 +
           i2 * m_stride.S2 +
           i3 * m_stride.S3 +
           i4 * m_stride.S4 +
           i5 * m_stride.S5 ;
  }

  //rank 7
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6 ) const
  {
    return i0 * m_stride.S0 +
           i1 * m_stride.S1 +
           i2 * m_stride.S2 +
           i3 * m_stride.S3 +
           i4 * m_stride.S4 +
           i5 * m_stride.S5 +
           i6 * m_stride.S6 ;
  }

  //rank 8
  template < typename I0, typename I1, typename I2, typename I3
           , typename I4, typename I5, typename I6, typename I7 >
  KOKKOS_INLINE_FUNCTION constexpr
  size_type operator()( I0 const & i0, I1 const & i1, I2 const & i2, I3 const & i3
                      , I4 const & i4, I5 const & i5, I6 const & i6, I7 const & i7 ) const
  {
    return i0 * m_stride.S0 +
           i1 * m_stride.S1 +
           i2 * m_stride.S2 +
           i3 * m_stride.S3 +
           i4 * m_stride.S4 +
           i5 * m_stride.S5 +
           i6 * m_stride.S6 +
           i7 * m_stride.S7 ;
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr array_layout layout() const
    {
      return array_layout( m_dim.N0 , m_stride.S0
                         , m_dim.N1 , m_stride.S1
                         , m_dim.N2 , m_stride.S2
                         , m_dim.N3 , m_stride.S3
                         , m_dim.N4 , m_stride.S4
                         , m_dim.N5 , m_stride.S5
                         , m_dim.N6 , m_stride.S6
                         , m_dim.N7 , m_stride.S7
                         );
    }

  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const { return m_dim.N0 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const { return m_dim.N1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const { return m_dim.N2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const { return m_dim.N3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const { return m_dim.N4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const { return m_dim.N5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const { return m_dim.N6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const { return m_dim.N7 ; }

  /* Cardinality of the domain index space */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type size() const
    { return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7 ; }

private:

  KOKKOS_INLINE_FUNCTION
  static constexpr size_type Max( size_type lhs , size_type rhs )
    { return lhs < rhs ? rhs : lhs ; }

public:

  /* Span of the range space, largest stride * dimension */
  KOKKOS_INLINE_FUNCTION
  constexpr size_type span() const
    {
      return Max( m_dim.N0 * m_stride.S0 ,
             Max( m_dim.N1 * m_stride.S1 ,
             Max( m_dim.N2 * m_stride.S2 ,
             Max( m_dim.N3 * m_stride.S3 ,
             Max( m_dim.N4 * m_stride.S4 ,
             Max( m_dim.N5 * m_stride.S5 ,
             Max( m_dim.N6 * m_stride.S6 ,
                  m_dim.N7 * m_stride.S7 )))))));
    }

  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const { return span() == size(); }

  /* Strides of dimensions */
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const { return m_stride.S0 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const { return m_stride.S1 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const { return m_stride.S2 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const { return m_stride.S3 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const { return m_stride.S4 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const { return m_stride.S5 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const { return m_stride.S6 ; }
  KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const { return m_stride.S7 ; }

  // Stride with [ rank ] value is the total length
  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  void stride( iType * const s ) const
    {
      if ( 0 < dimension_type::rank ) { s[0] = m_stride.S0 ; }
      if ( 1 < dimension_type::rank ) { s[1] = m_stride.S1 ; }
      if ( 2 < dimension_type::rank ) { s[2] = m_stride.S2 ; }
      if ( 3 < dimension_type::rank ) { s[3] = m_stride.S3 ; }
      if ( 4 < dimension_type::rank ) { s[4] = m_stride.S4 ; }
      if ( 5 < dimension_type::rank ) { s[5] = m_stride.S5 ; }
      if ( 6 < dimension_type::rank ) { s[6] = m_stride.S6 ; }
      if ( 7 < dimension_type::rank ) { s[7] = m_stride.S7 ; }
      s[dimension_type::rank] = span();
    }

  //----------------------------------------

  ViewOffset() = default ;
  ViewOffset( const ViewOffset & ) = default ;
  ViewOffset & operator = ( const ViewOffset & ) = default ;

  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( std::integral_constant<unsigned,0> const &
                      , Kokkos::LayoutStride const & rhs )
    : m_dim( rhs.dimension[0] , rhs.dimension[1] , rhs.dimension[2] , rhs.dimension[3]
           , rhs.dimension[4] , rhs.dimension[5] , rhs.dimension[6] , rhs.dimension[7] )
    , m_stride( rhs.stride[0] , rhs.stride[1] , rhs.stride[2] , rhs.stride[3]
              , rhs.stride[4] , rhs.stride[5] , rhs.stride[6] , rhs.stride[7] )
    {}

  template< class DimRHS , class LayoutRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset( const ViewOffset< DimRHS , LayoutRHS , void > & rhs )
    : m_dim( rhs.m_dim.N0 , rhs.m_dim.N1 , rhs.m_dim.N2 , rhs.m_dim.N3
           , rhs.m_dim.N4 , rhs.m_dim.N5 , rhs.m_dim.N6 , rhs.m_dim.N7 )
    , m_stride( rhs.stride_0() , rhs.stride_1() , rhs.stride_2() , rhs.stride_3()
              , rhs.stride_4() , rhs.stride_5() , rhs.stride_6() , rhs.stride_7() )
    {
      static_assert( int(DimRHS::rank) == int(dimension_type::rank) , "ViewOffset assignment requires equal rank" );
      // Also requires equal static dimensions ...
    }

  //----------------------------------------
  // Subview construction

private:

  template< class DimRHS , class LayoutRHS >
  KOKKOS_INLINE_FUNCTION static
  constexpr size_t stride
    ( unsigned r , const ViewOffset< DimRHS , LayoutRHS , void > & rhs )
    {
      return r >  7 ? 0 : (
             r == 0 ? rhs.stride_0() : (
             r == 1 ? rhs.stride_1() : (
             r == 2 ? rhs.stride_2() : (
             r == 3 ? rhs.stride_3() : (
             r == 4 ? rhs.stride_4() : (
             r == 5 ? rhs.stride_5() : (
             r == 6 ? rhs.stride_6() : rhs.stride_7() )))))));
    }

public:

  template< class DimRHS , class LayoutRHS >
  KOKKOS_INLINE_FUNCTION
  constexpr ViewOffset
    ( const ViewOffset< DimRHS , LayoutRHS , void > & rhs
    , const SubviewExtents< DimRHS::rank , dimension_type::rank > & sub
    )
    // range_extent(r) returns 0 when dimension_type::rank <= r
    : m_dim( sub.range_extent(0)
           , sub.range_extent(1)
           , sub.range_extent(2)
           , sub.range_extent(3)
           , sub.range_extent(4)
           , sub.range_extent(5)
           , sub.range_extent(6)
           , sub.range_extent(7)
           )
    // range_index(r) returns ~0u when dimension_type::rank <= r
    , m_stride( stride( sub.range_index(0), rhs )
              , stride( sub.range_index(1), rhs )
              , stride( sub.range_index(2), rhs )
              , stride( sub.range_index(3), rhs )
              , stride( sub.range_index(4), rhs )
              , stride( sub.range_index(5), rhs )
              , stride( sub.range_index(6), rhs )
              , stride( sub.range_index(7), rhs )
              )
    {}
};

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  ViewDataHandle provides the type of the 'data handle' which the view
 *          uses to access data with the [] operator. It also provides
 *          an allocate function and a function to extract a raw ptr from the
 *          data handle. ViewDataHandle also defines an enum ReferenceAble which
 *          specifies whether references/pointers to elements can be taken and a
 *          'return_type' which is what the view operators will give back.
 *          Specialisation of this object allows three things depending
 *          on ViewTraits and compiler options:
 *          (i)   Use special allocator (e.g. huge pages/small pages and pinned memory)
 *          (ii)  Use special data handle type (e.g. add Cuda Texture Object)
 *          (iii) Use special access intrinsics (e.g. texture fetch and non-caching loads)
 */
template< class Traits , class Enable = void >
struct ViewDataHandle {

  typedef typename Traits::value_type   value_type  ;
  typedef typename Traits::value_type * handle_type ;
  typedef typename Traits::value_type & return_type ;
  typedef Kokkos::Impl::SharedAllocationTracker  track_type  ;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( value_type * arg_data_ptr
                           , track_type const & /*arg_tracker*/ )
  {
    return handle_type( arg_data_ptr );
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( handle_type const arg_data_ptr
                           , size_t offset )
  {
    return handle_type( arg_data_ptr + offset );
  }
};

template< class Traits >
struct ViewDataHandle< Traits ,
  typename std::enable_if<( std::is_same< typename Traits::non_const_value_type
                                        , typename Traits::value_type >::value
                            &&
                            std::is_same< typename Traits::specialize , void >::value
                            &&
                            Traits::memory_traits::Atomic
                          )>::type >
{
  typedef typename Traits::value_type  value_type ;
  typedef typename Kokkos::Impl::AtomicViewDataHandle< Traits >  handle_type ;
  typedef typename Kokkos::Impl::AtomicDataElement< Traits >     return_type ;
  typedef Kokkos::Impl::SharedAllocationTracker                  track_type  ;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( value_type * arg_data_ptr
                           , track_type const & /*arg_tracker*/ )
  {
    return handle_type( arg_data_ptr );
  }

  template<class SrcHandleType>
  KOKKOS_INLINE_FUNCTION
  static handle_type assign( const SrcHandleType& arg_handle
                           , size_t offset )
  {
    return handle_type( arg_handle.ptr + offset );
  }
};

template< class Traits >
struct ViewDataHandle< Traits ,
  typename std::enable_if<(
                            std::is_same< typename Traits::specialize , void >::value
                            &&
                            (!Traits::memory_traits::Aligned)
                            &&
                            Traits::memory_traits::Restrict
#ifdef KOKKOS_ENABLE_CUDA
                            &&
                            (!( std::is_same< typename Traits::memory_space,Kokkos::CudaSpace>::value ||
                                std::is_same< typename Traits::memory_space,Kokkos::CudaUVMSpace>::value ))
#endif
                            &&
                            (!Traits::memory_traits::Atomic)
                          )>::type >
{
  typedef typename Traits::value_type  value_type ;
  typedef typename Traits::value_type * KOKKOS_RESTRICT handle_type ;
  typedef typename Traits::value_type & KOKKOS_RESTRICT return_type ;
  typedef Kokkos::Impl::SharedAllocationTracker  track_type  ;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( value_type * arg_data_ptr
                           , track_type const & /*arg_tracker*/ )
  {
    return handle_type( arg_data_ptr );
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( handle_type const arg_data_ptr
                           , size_t offset )
  {
    return handle_type( arg_data_ptr + offset );
  }
};

template< class Traits >
struct ViewDataHandle< Traits ,
  typename std::enable_if<(
                            std::is_same< typename Traits::specialize , void >::value
                            &&
                            Traits::memory_traits::Aligned
			    &&
                            (!Traits::memory_traits::Restrict)
#ifdef KOKKOS_ENABLE_CUDA
                            &&
                            (!( std::is_same< typename Traits::memory_space,Kokkos::CudaSpace>::value ||
                                std::is_same< typename Traits::memory_space,Kokkos::CudaUVMSpace>::value ))
#endif
                            &&
                            (!Traits::memory_traits::Atomic)
                          )>::type >
{
  typedef typename Traits::value_type  value_type ;
  typedef typename Traits::value_type * KOKKOS_IMPL_ALIGN_PTR(KOKKOS_MEMORY_ALIGNMENT) handle_type ;
  typedef typename Traits::value_type & return_type ;
  typedef Kokkos::Impl::SharedAllocationTracker  track_type  ;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( value_type * arg_data_ptr
                           , track_type const & /*arg_tracker*/ )
  {
    if ( reinterpret_cast<uintptr_t>(arg_data_ptr) % Impl::MEMORY_ALIGNMENT ) {
      Kokkos::abort("Assigning NonAligned View or Pointer to Kokkos::View with Aligned attribute");
    }
    return handle_type( arg_data_ptr );
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( handle_type const arg_data_ptr
                           , size_t offset )
  {
    if ( reinterpret_cast<uintptr_t>(arg_data_ptr+offset) % Impl::MEMORY_ALIGNMENT ) {
      Kokkos::abort("Assigning NonAligned View or Pointer to Kokkos::View with Aligned attribute");
    }
    return handle_type( arg_data_ptr + offset );
  }
};

template< class Traits >
struct ViewDataHandle< Traits ,
  typename std::enable_if<(
                            std::is_same< typename Traits::specialize , void >::value
                            &&
                            Traits::memory_traits::Aligned
                            &&
                            Traits::memory_traits::Restrict
#ifdef KOKKOS_ENABLE_CUDA
                            &&
                            (!( std::is_same< typename Traits::memory_space,Kokkos::CudaSpace>::value ||
                                std::is_same< typename Traits::memory_space,Kokkos::CudaUVMSpace>::value ))
#endif
                            &&
                            (!Traits::memory_traits::Atomic)
                          )>::type >
{
  typedef typename Traits::value_type  value_type ;
  typedef typename Traits::value_type * KOKKOS_RESTRICT KOKKOS_IMPL_ALIGN_PTR(KOKKOS_MEMORY_ALIGNMENT) handle_type ;
  typedef typename Traits::value_type & return_type ;
  typedef Kokkos::Impl::SharedAllocationTracker  track_type  ;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( value_type * arg_data_ptr
                           , track_type const & /*arg_tracker*/ )
  {
    if ( reinterpret_cast<uintptr_t>(arg_data_ptr) % Impl::MEMORY_ALIGNMENT ) {
      Kokkos::abort("Assigning NonAligned View or Pointer to Kokkos::View with Aligned attribute");
    }
    return handle_type( arg_data_ptr );
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign( handle_type const arg_data_ptr
                           , size_t offset )
  {
    if ( reinterpret_cast<uintptr_t>(arg_data_ptr+offset) % Impl::MEMORY_ALIGNMENT ) {
      Kokkos::abort("Assigning NonAligned View or Pointer to Kokkos::View with Aligned attribute");
    }
    return handle_type( arg_data_ptr + offset );
  }
};
}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------

/*
 *  The construction, assignment to default, and destruction
 *  are merged into a single functor.
 *  Primarily to work around an unresolved CUDA back-end bug
 *  that would lose the destruction cuda device function when
 *  called from the shared memory tracking destruction.
 *  Secondarily to have two fewer partial specializations.
 */
template< class ExecSpace
        , class ValueType
        , bool IsScalar = std::is_scalar< ValueType >::value
        >
struct ViewValueFunctor ;

template< class ExecSpace , class ValueType >
struct ViewValueFunctor< ExecSpace , ValueType , false /* is_scalar */ >
{
  typedef Kokkos::RangePolicy< ExecSpace > PolicyType ;
  typedef typename ExecSpace::execution_space Exec;

  Exec        space ;
  ValueType * ptr ;
  size_t      n ;
  bool        destroy ;

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_t i ) const
    {
      if ( destroy ) { (ptr+i)->~ValueType(); } //KOKKOS_IMPL_CUDA_CLANG_WORKAROUND this line causes ptax error __cxa_begin_catch in nested_view unit-test
      else           { new (ptr+i) ValueType(); }
    }

  ViewValueFunctor() = default ;
  ViewValueFunctor( const ViewValueFunctor & ) = default ;
  ViewValueFunctor & operator = ( const ViewValueFunctor & ) = default ;

  ViewValueFunctor( ExecSpace   const & arg_space
                  , ValueType * const arg_ptr
                  , size_t      const arg_n )
    : space( arg_space )
    , ptr( arg_ptr )
    , n( arg_n )
    , destroy( false )
    {}

  void execute( bool arg )
    {
      destroy = arg ;
      if ( ! space.in_parallel() ) {
#if defined(KOKKOS_ENABLE_PROFILING)
        uint64_t kpID = 0;
        if(Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::beginParallelFor("Kokkos::View::initialization", 0, &kpID);
        }
#endif
        const Kokkos::Impl::ParallelFor< ViewValueFunctor , PolicyType >
          closure( *this , PolicyType( 0 , n ) );
        closure.execute();
        space.fence();
#if defined(KOKKOS_ENABLE_PROFILING)
        if(Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::endParallelFor(kpID);
        }
#endif
      }
      else {
        for ( size_t i = 0 ; i < n ; ++i ) operator()(i);
      }
    }

  void construct_shared_allocation()
    { execute( false ); }

  void destroy_shared_allocation()
    { execute( true ); }
};


template< class ExecSpace , class ValueType >
struct ViewValueFunctor< ExecSpace , ValueType , true /* is_scalar */ >
{
  typedef Kokkos::RangePolicy< ExecSpace > PolicyType ;

  ExecSpace   space ;
  ValueType * ptr ;
  size_t      n ;

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_t i ) const
    { ptr[i] = ValueType(); }

  ViewValueFunctor() = default ;
  ViewValueFunctor( const ViewValueFunctor & ) = default ;
  ViewValueFunctor & operator = ( const ViewValueFunctor & ) = default ;

  ViewValueFunctor( ExecSpace   const & arg_space
                  , ValueType * const arg_ptr
                  , size_t      const arg_n )
    : space( arg_space )
    , ptr( arg_ptr )
    , n( arg_n )
    {}

  void construct_shared_allocation()
    {
      if ( ! space.in_parallel() ) {
#if defined(KOKKOS_ENABLE_PROFILING)
        uint64_t kpID = 0;
        if(Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::beginParallelFor("Kokkos::View::initialization", 0, &kpID);
        }
#endif
        const Kokkos::Impl::ParallelFor< ViewValueFunctor , PolicyType >
          closure( *this , PolicyType( 0 , n ) );
        closure.execute();
        space.fence();
#if defined(KOKKOS_ENABLE_PROFILING)
        if(Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::endParallelFor(kpID);
        }
#endif
      }
      else {
        for ( size_t i = 0 ; i < n ; ++i ) operator()(i);
      }
    }

  void destroy_shared_allocation() {}
};

//----------------------------------------------------------------------------
/** \brief  View mapping for non-specialized data type and standard layout */
template< class Traits >
class ViewMapping< Traits ,
  typename std::enable_if<(
    std::is_same< typename Traits::specialize , void >::value
    &&
    ViewOffset< typename Traits::dimension
              , typename Traits::array_layout
              , void >::is_mapping_plugin::value
  )>::type >
{
private:

  template< class , class ... > friend class ViewMapping ;
  template< class , class ... > friend class Kokkos::View ;

  typedef ViewOffset< typename Traits::dimension
                    , typename Traits::array_layout
                    , void
                    >  offset_type ;

  typedef typename ViewDataHandle< Traits >::handle_type  handle_type ;

  handle_type  m_handle ;
  offset_type  m_offset ;

  KOKKOS_INLINE_FUNCTION
  ViewMapping( const handle_type & arg_handle , const offset_type & arg_offset )
    : m_handle( arg_handle )
    , m_offset( arg_offset )
    {}

public:

  typedef void printable_label_typedef;
  enum { is_managed = Traits::is_managed };

  //----------------------------------------
  // Domain dimensions

  enum { Rank = Traits::dimension::rank };

  template< typename iType >
  KOKKOS_INLINE_FUNCTION constexpr size_t extent( const iType & r ) const
    { return m_offset.m_dim.extent(r); }

  KOKKOS_INLINE_FUNCTION constexpr
  typename Traits::array_layout layout() const
    { return m_offset.layout(); }

  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0() const { return m_offset.dimension_0(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_1() const { return m_offset.dimension_1(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_2() const { return m_offset.dimension_2(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_3() const { return m_offset.dimension_3(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_4() const { return m_offset.dimension_4(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_5() const { return m_offset.dimension_5(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_6() const { return m_offset.dimension_6(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_7() const { return m_offset.dimension_7(); }

  // Is a regular layout with uniform striding for each index.
  using is_regular = typename offset_type::is_regular ;

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const { return m_offset.stride_0(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const { return m_offset.stride_1(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const { return m_offset.stride_2(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const { return m_offset.stride_3(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const { return m_offset.stride_4(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const { return m_offset.stride_5(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const { return m_offset.stride_6(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const { return m_offset.stride_7(); }

  template< typename iType >
  KOKKOS_INLINE_FUNCTION void stride( iType * const s ) const { m_offset.stride(s); }

  //----------------------------------------
  // Range span

  /** \brief  Span of the mapped range */
  KOKKOS_INLINE_FUNCTION constexpr size_t span() const { return m_offset.span(); }

  /** \brief  Is the mapped range span contiguous */
  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const { return m_offset.span_is_contiguous(); }

  typedef typename ViewDataHandle< Traits >::return_type  reference_type ;
  typedef typename Traits::value_type *                   pointer_type ;

  /** \brief  Query raw pointer to memory */
  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const
    {
      return m_handle;
    }

  //----------------------------------------
  // The View class performs all rank and bounds checking before
  // calling these element reference methods.

  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference() const { return m_handle[0]; }

  template< typename I0 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename
    std::enable_if< std::is_integral<I0>::value &&
                    ! std::is_same< typename Traits::array_layout , Kokkos::LayoutStride >::value
                  , reference_type >::type
  reference( const I0 & i0 ) const { return m_handle[i0]; }

  template< typename I0 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename
    std::enable_if< std::is_integral<I0>::value &&
                    std::is_same< typename Traits::array_layout , Kokkos::LayoutStride >::value
                  , reference_type >::type
  reference( const I0 & i0 ) const { return m_handle[ m_offset(i0) ]; }

  template< typename I0 , typename I1 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 ) const
    { return m_handle[ m_offset(i0,i1) ]; }

  template< typename I0 , typename I1 , typename I2 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 ) const
    { return m_handle[ m_offset(i0,i1,i2) ]; }

  template< typename I0 , typename I1 , typename I2 , typename I3 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3 ) const
    { return m_handle[ m_offset(i0,i1,i2,i3) ]; }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 ) const
    { return m_handle[ m_offset(i0,i1,i2,i3,i4) ]; }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 , const I5 & i5 ) const
    { return m_handle[ m_offset(i0,i1,i2,i3,i4,i5) ]; }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 , typename I6 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 , const I5 & i5 , const I6 & i6 ) const
    { return m_handle[ m_offset(i0,i1,i2,i3,i4,i5,i6) ]; }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 , typename I6 , typename I7 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
                          , const I4 & i4 , const I5 & i5 , const I6 & i6 , const I7 & i7 ) const
    { return m_handle[ m_offset(i0,i1,i2,i3,i4,i5,i6,i7) ]; }

  //----------------------------------------

private:

  enum { MemorySpanMask = 8 - 1 /* Force alignment on 8 byte boundary */ };
  enum { MemorySpanSize = sizeof(typename Traits::value_type) };

public:

  /** \brief  Span, in bytes, of the referenced memory */
  KOKKOS_INLINE_FUNCTION constexpr size_t memory_span() const
    {
      return ( m_offset.span() * sizeof(typename Traits::value_type) + MemorySpanMask ) & ~size_t(MemorySpanMask);
    }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION ~ViewMapping() {}
  KOKKOS_INLINE_FUNCTION ViewMapping() : m_handle(), m_offset() {}
  KOKKOS_INLINE_FUNCTION ViewMapping( const ViewMapping & rhs )
    : m_handle( rhs.m_handle ), m_offset( rhs.m_offset ) {}
  KOKKOS_INLINE_FUNCTION ViewMapping & operator = ( const ViewMapping & rhs )
    { m_handle = rhs.m_handle ; m_offset = rhs.m_offset ; return *this ; }

  KOKKOS_INLINE_FUNCTION ViewMapping( ViewMapping && rhs )
    : m_handle( rhs.m_handle ), m_offset( rhs.m_offset ) {}
  KOKKOS_INLINE_FUNCTION ViewMapping & operator = ( ViewMapping && rhs )
    { m_handle = rhs.m_handle ; m_offset = rhs.m_offset ; return *this ; }

  //----------------------------------------

  /**\brief  Span, in bytes, of the required memory */
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t memory_span( typename Traits::array_layout const & arg_layout )
    {
      typedef std::integral_constant< unsigned , 0 >  padding ;
      return ( offset_type( padding(), arg_layout ).span() * MemorySpanSize + MemorySpanMask ) & ~size_t(MemorySpanMask);
    }

  /**\brief  Wrap a span of memory */
  template< class ... P >
  KOKKOS_INLINE_FUNCTION
  ViewMapping( Kokkos::Impl::ViewCtorProp< P ... > const & arg_prop
             , typename Traits::array_layout const & arg_layout
             )
    : m_handle( ( (Kokkos::Impl::ViewCtorProp<void,pointer_type> const &) arg_prop ).value )
    , m_offset( std::integral_constant< unsigned , 0 >() , arg_layout )
    {}

  /**\brief  Assign data */
  KOKKOS_INLINE_FUNCTION
  void assign_data( pointer_type arg_ptr )
    { m_handle = handle_type( arg_ptr ); }

  //----------------------------------------
  /*  Allocate and construct mapped array.
   *  Allocate via shared allocation record and
   *  return that record for allocation tracking.
   */
  template< class ... P >
  Kokkos::Impl::SharedAllocationRecord<> *
  allocate_shared( Kokkos::Impl::ViewCtorProp< P... > const & arg_prop
                 , typename Traits::array_layout const & arg_layout )
  {
    typedef Kokkos::Impl::ViewCtorProp< P... > alloc_prop ;

    typedef typename alloc_prop::execution_space  execution_space ;
    typedef typename Traits::memory_space         memory_space ;
    typedef typename Traits::value_type           value_type ;
    typedef ViewValueFunctor< execution_space , value_type > functor_type ;
    typedef Kokkos::Impl::SharedAllocationRecord< memory_space , functor_type > record_type ;

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    typedef std::integral_constant
      < unsigned
      , alloc_prop::allow_padding ? sizeof(value_type) : 0
      > padding ;

    m_offset = offset_type( padding(), arg_layout );

    const size_t alloc_size =
      ( m_offset.span() * MemorySpanSize + MemorySpanMask ) & ~size_t(MemorySpanMask);

    // Create shared memory tracking record with allocate memory from the memory space
    record_type * const record =
      record_type::allocate( ( (Kokkos::Impl::ViewCtorProp<void,memory_space> const &) arg_prop ).value
                           , ( (Kokkos::Impl::ViewCtorProp<void,std::string>  const &) arg_prop ).value
                           , alloc_size );

    //  Only set the the pointer and initialize if the allocation is non-zero.
    //  May be zero if one of the dimensions is zero.
    if ( alloc_size ) {

      m_handle = handle_type( reinterpret_cast< pointer_type >( record->data() ) );

      if ( alloc_prop::initialize ) {
        // Assume destruction is only required when construction is requested.
        // The ViewValueFunctor has both value construction and destruction operators.
        record->m_destroy = functor_type( ( (Kokkos::Impl::ViewCtorProp<void,execution_space> const &) arg_prop).value
                                        , (value_type *) m_handle
                                        , m_offset.span()
                                        );

        // Construct values
        record->m_destroy.construct_shared_allocation();
      }
    }

    return record ;
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/** \brief  Assign compatible default mappings */

template< class DstTraits , class SrcTraits >
class ViewMapping< DstTraits , SrcTraits ,
  typename std::enable_if<(
    /* default mappings */
    std::is_same< typename DstTraits::specialize , void >::value
    &&
    std::is_same< typename SrcTraits::specialize , void >::value
    &&
    (
      /* same layout */
      std::is_same< typename DstTraits::array_layout , typename SrcTraits::array_layout >::value
      ||
      /* known layout */
      (
        (
          std::is_same< typename DstTraits::array_layout , Kokkos::LayoutLeft >::value ||
          std::is_same< typename DstTraits::array_layout , Kokkos::LayoutRight >::value ||
          std::is_same< typename DstTraits::array_layout , Kokkos::LayoutStride >::value
        )
        &&
        (
          std::is_same< typename SrcTraits::array_layout , Kokkos::LayoutLeft >::value ||
          std::is_same< typename SrcTraits::array_layout , Kokkos::LayoutRight >::value ||
          std::is_same< typename SrcTraits::array_layout , Kokkos::LayoutStride >::value
        )
      )
    )
  )>::type >
{
private:

  enum { is_assignable_space =
#if 1
   Kokkos::Impl::MemorySpaceAccess
     < typename DstTraits::memory_space
     , typename SrcTraits::memory_space >::assignable };
#else
   std::is_same< typename DstTraits::memory_space
               , typename SrcTraits::memory_space >::value };
#endif

  enum { is_assignable_value_type =
    std::is_same< typename DstTraits::value_type
                , typename SrcTraits::value_type >::value ||
    std::is_same< typename DstTraits::value_type
                , typename SrcTraits::const_value_type >::value };

  enum { is_assignable_dimension =
    ViewDimensionAssignable< typename DstTraits::dimension
                           , typename SrcTraits::dimension >::value };

  enum { is_assignable_layout =
    std::is_same< typename DstTraits::array_layout
                , typename SrcTraits::array_layout >::value ||
    std::is_same< typename DstTraits::array_layout
                , Kokkos::LayoutStride >::value ||
    ( DstTraits::dimension::rank == 0 ) ||
    ( DstTraits::dimension::rank == 1 &&
      DstTraits::dimension::rank_dynamic == 1 )
    };

public:

  enum { is_assignable = is_assignable_space &&
                         is_assignable_value_type &&
                         is_assignable_dimension &&
                         is_assignable_layout };

  typedef Kokkos::Impl::SharedAllocationTracker  TrackType ;
  typedef ViewMapping< DstTraits , void >  DstType ;
  typedef ViewMapping< SrcTraits , void >  SrcType ;

  KOKKOS_INLINE_FUNCTION
  static void assign( DstType & dst , const SrcType & src , const TrackType & src_track )
    {
      static_assert( is_assignable_space
                   , "View assignment must have compatible spaces" );

      static_assert( is_assignable_value_type
                   , "View assignment must have same value type or const = non-const" );

      static_assert( is_assignable_dimension
                   , "View assignment must have compatible dimensions" );

      static_assert( is_assignable_layout
                   , "View assignment must have compatible layout or have rank <= 1" );

      typedef typename DstType::offset_type  dst_offset_type ;

      if ( size_t(DstTraits::dimension::rank_dynamic) < size_t(SrcTraits::dimension::rank_dynamic) ) {
        typedef typename DstTraits::dimension dst_dim;
        bool assignable =
          ( ( 1 > DstTraits::dimension::rank_dynamic && 1 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN0 == src.dimension_0() : true ) &&
          ( ( 2 > DstTraits::dimension::rank_dynamic && 2 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN1 == src.dimension_1() : true ) &&
          ( ( 3 > DstTraits::dimension::rank_dynamic && 3 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN2 == src.dimension_2() : true ) &&
          ( ( 4 > DstTraits::dimension::rank_dynamic && 4 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN3 == src.dimension_3() : true ) &&
          ( ( 5 > DstTraits::dimension::rank_dynamic && 5 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN4 == src.dimension_4() : true ) &&
          ( ( 6 > DstTraits::dimension::rank_dynamic && 6 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN5 == src.dimension_5() : true ) &&
          ( ( 7 > DstTraits::dimension::rank_dynamic && 7 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN6 == src.dimension_6() : true ) &&
          ( ( 8 > DstTraits::dimension::rank_dynamic && 8 <= SrcTraits::dimension::rank_dynamic ) ?
            dst_dim::ArgN7 == src.dimension_7() : true )
          ;
        if(!assignable)
          Kokkos::abort("View Assignment: trying to assign runtime dimension to non matching compile time dimension.");
      }
      dst.m_offset = dst_offset_type( src.m_offset );
      dst.m_handle = Kokkos::Impl::ViewDataHandle< DstTraits >::assign( src.m_handle , src_track );
    }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Subview mapping.
// Deduce destination view type from source view traits and subview arguments

template< class SrcTraits , class ... Args >
struct ViewMapping
  < typename std::enable_if<(
      std::is_same< typename SrcTraits::specialize , void >::value
      &&
      (
        std::is_same< typename SrcTraits::array_layout
                    , Kokkos::LayoutLeft >::value ||
        std::is_same< typename SrcTraits::array_layout
                    , Kokkos::LayoutRight >::value ||
        std::is_same< typename SrcTraits::array_layout
                    , Kokkos::LayoutStride >::value
      )
    )>::type
  , SrcTraits
  , Args ... >
{
private:

  static_assert( SrcTraits::rank == sizeof...(Args) ,
    "Subview mapping requires one argument for each dimension of source View" );

  enum
    { RZ = false
    , R0 = bool(is_integral_extent<0,Args...>::value)
    , R1 = bool(is_integral_extent<1,Args...>::value)
    , R2 = bool(is_integral_extent<2,Args...>::value)
    , R3 = bool(is_integral_extent<3,Args...>::value)
    , R4 = bool(is_integral_extent<4,Args...>::value)
    , R5 = bool(is_integral_extent<5,Args...>::value)
    , R6 = bool(is_integral_extent<6,Args...>::value)
    , R7 = bool(is_integral_extent<7,Args...>::value)
    };

  enum { rank = unsigned(R0) + unsigned(R1) + unsigned(R2) + unsigned(R3)
              + unsigned(R4) + unsigned(R5) + unsigned(R6) + unsigned(R7) };

  // Whether right-most rank is a range.
  enum { R0_rev = ( 0 == SrcTraits::rank ? RZ : (
                    1 == SrcTraits::rank ? R0 : (
                    2 == SrcTraits::rank ? R1 : (
                    3 == SrcTraits::rank ? R2 : (
                    4 == SrcTraits::rank ? R3 : (
                    5 == SrcTraits::rank ? R4 : (
                    6 == SrcTraits::rank ? R5 : (
                    7 == SrcTraits::rank ? R6 : R7 )))))))) };

  // Subview's layout
  typedef typename std::conditional<
      ( /* Same array layout IF */
        ( rank == 0 ) /* output rank zero */
        ||
        SubviewLegalArgsCompileTime<typename SrcTraits::array_layout, typename SrcTraits::array_layout,
                                    rank, SrcTraits::rank, 0, Args...>::value
        ||
        // OutputRank 1 or 2, InputLayout Left, Interval 0
        // because single stride one or second index has a stride.
        ( rank <= 2 && R0 && std::is_same< typename SrcTraits::array_layout , Kokkos::LayoutLeft >::value ) //replace with input rank
        ||
        // OutputRank 1 or 2, InputLayout Right, Interval [InputRank-1]
        // because single stride one or second index has a stride.
        ( rank <= 2 && R0_rev && std::is_same< typename SrcTraits::array_layout , Kokkos::LayoutRight >::value ) //replace input rank
      ), typename SrcTraits::array_layout , Kokkos::LayoutStride
      >::type array_layout ;

  typedef typename SrcTraits::value_type  value_type ;

  typedef typename std::conditional< rank == 0 , value_type ,
          typename std::conditional< rank == 1 , value_type * ,
          typename std::conditional< rank == 2 , value_type ** ,
          typename std::conditional< rank == 3 , value_type *** ,
          typename std::conditional< rank == 4 , value_type **** ,
          typename std::conditional< rank == 5 , value_type ***** ,
          typename std::conditional< rank == 6 , value_type ****** ,
          typename std::conditional< rank == 7 , value_type ******* ,
                                                 value_type ********
          >::type >::type >::type >::type >::type >::type >::type >::type
     data_type ;

public:

  typedef Kokkos::ViewTraits
    < data_type
    , array_layout
    , typename SrcTraits::device_type
    , typename SrcTraits::memory_traits > traits_type ;

  typedef Kokkos::View
    < data_type
    , array_layout
    , typename SrcTraits::device_type
    , typename SrcTraits::memory_traits > type ;

  template< class MemoryTraits >
  struct apply {

    static_assert( Kokkos::Impl::is_memory_traits< MemoryTraits >::value , "" );

    typedef Kokkos::ViewTraits
      < data_type
      , array_layout
      , typename SrcTraits::device_type
      , MemoryTraits > traits_type ;

    typedef Kokkos::View
      < data_type
      , array_layout
      , typename SrcTraits::device_type
      , MemoryTraits > type ;
  };

  // The presumed type is 'ViewMapping< traits_type , void >'
  // However, a compatible ViewMapping is acceptable.
  template< class DstTraits >
  KOKKOS_INLINE_FUNCTION
  static void assign( ViewMapping< DstTraits , void > & dst
                    , ViewMapping< SrcTraits , void > const & src
                    , Args ... args )
    {
      static_assert(
        ViewMapping< DstTraits , traits_type , void >::is_assignable ,
        "Subview destination type must be compatible with subview derived type" );

      typedef ViewMapping< DstTraits , void >  DstType ;

      typedef typename DstType::offset_type  dst_offset_type ;

      const SubviewExtents< SrcTraits::rank , rank >
        extents( src.m_offset.m_dim , args... );

      dst.m_offset = dst_offset_type( src.m_offset , extents );

      dst.m_handle = ViewDataHandle< DstTraits >::assign(src.m_handle,
          src.m_offset( extents.domain_offset(0)
                      , extents.domain_offset(1)
                      , extents.domain_offset(2)
                      , extents.domain_offset(3)
                      , extents.domain_offset(4)
                      , extents.domain_offset(5)
                      , extents.domain_offset(6)
                      , extents.domain_offset(7)
          ));
    }
};



//----------------------------------------------------------------------------

}} // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template< unsigned , class MapType >
KOKKOS_INLINE_FUNCTION
bool view_verify_operator_bounds( const MapType & )
{ return true ; }

template< unsigned R , class MapType , class iType , class ... Args >
KOKKOS_INLINE_FUNCTION
bool view_verify_operator_bounds
  ( const MapType & map
  , const iType   & i
  , Args ... args
  )
{
  return ( size_t(i) < map.extent(R) )
         && view_verify_operator_bounds<R+1>( map , args ... );
}

template< unsigned , class MapType >
inline
void view_error_operator_bounds( char * , int , const MapType & )
{}

template< unsigned R , class MapType , class iType , class ... Args >
inline
void view_error_operator_bounds
  ( char * buf
  , int len
  , const MapType & map
  , const iType   & i
  , Args ... args
  )
{
  const int n =
    snprintf(buf,len," %ld < %ld %c"
            , static_cast<unsigned long>(i)
            , static_cast<unsigned long>( map.extent(R) )
            , ( sizeof...(Args) ? ',' : ')' )
            );
  view_error_operator_bounds<R+1>(buf+n,len-n,map,args...);
}

#if ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )

/* Check #3: is the View managed as determined by the MemoryTraits? */
template< class MapType,
  bool is_managed = (MapType::is_managed != 0) >
struct OperatorBoundsErrorOnDevice;

template< class MapType >
struct OperatorBoundsErrorOnDevice< MapType, false > {
KOKKOS_INLINE_FUNCTION
static void run(MapType const&) {
  Kokkos::abort("View bounds error");
}
};

template< class MapType >
struct OperatorBoundsErrorOnDevice< MapType, true > {
KOKKOS_INLINE_FUNCTION
static void run(MapType const& map) {
  char const* const user_alloc_start = reinterpret_cast<char const*>(map.data());
  char const* const header_start = user_alloc_start - sizeof(SharedAllocationHeader);
  SharedAllocationHeader const* const header =
    reinterpret_cast<SharedAllocationHeader const*>(header_start);
  char const* const label = header->label();
  enum { LEN = 128 };
  char msg[LEN];
  char const* const first_part = "View bounds error of view ";
  char* p = msg;
  char* const end = msg + LEN - 1;
  for (char const* p2 = first_part; (*p2 != '\0') && (p < end); ++p, ++p2) {
    *p = *p2;
  }
  for (char const* p2 = label; (*p2 != '\0') && (p < end); ++p, ++p2) {
    *p = *p2;
  }
  *p = '\0';
  Kokkos::abort(msg);
}
};

/* Check #2: does the ViewMapping have the printable_label_typedef defined?
   See above that only the non-specialized standard-layout ViewMapping has
   this defined by default.
   The existence of this typedef indicates the existence of MapType::is_managed */
template< class T, class Enable = void >
struct has_printable_label_typedef : public std::false_type {};

template<class T>
struct has_printable_label_typedef<
  T, typename enable_if_type<typename T::printable_label_typedef>::type>
  : public std::true_type
{};

template< class MapType >
KOKKOS_INLINE_FUNCTION
void operator_bounds_error_on_device(
    MapType const&,
    std::false_type) {
  Kokkos::abort("View bounds error");
}

template< class MapType >
KOKKOS_INLINE_FUNCTION
void operator_bounds_error_on_device(
    MapType const& map,
    std::true_type) {
  OperatorBoundsErrorOnDevice< MapType >::run(map);
}

#endif // ! defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )

template< class MemorySpace , class MapType , class ... Args >
KOKKOS_INLINE_FUNCTION
void view_verify_operator_bounds
  ( Kokkos::Impl::SharedAllocationTracker const & tracker
  , const MapType & map , Args ... args )
{
  if ( ! view_verify_operator_bounds<0>( map , args ... ) ) {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    enum { LEN = 1024 };
    char buffer[ LEN ];
    const std::string label = tracker.template get_label<MemorySpace>();
    int n = snprintf(buffer,LEN,"View bounds error of view %s (",label.c_str());
    view_error_operator_bounds<0>( buffer + n , LEN - n , map , args ... );
    Kokkos::Impl::throw_runtime_exception(std::string(buffer));
#else
    /* Check #1: is there a SharedAllocationRecord?
       (we won't use it, but if its not there then there isn't
        a corresponding SharedAllocationHeader containing a label).
       This check should cover the case of Views that don't
       have the Unmanaged trait but were initialized by pointer. */
    if (tracker.has_record()) {
      operator_bounds_error_on_device<MapType>(
          map, has_printable_label_typedef<MapType>());
    } else {
      Kokkos::abort("View bounds error");
    }
#endif
  }
}

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #ifndef KOKKOS_EXPERIMENTAL_VIEW_MAPPING_HPP */

