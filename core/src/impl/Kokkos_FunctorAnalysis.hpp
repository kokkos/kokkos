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

#ifndef KOKKOS_FUNCTORANALYSIS_HPP
#define KOKKOS_FUNCTORANALYSIS_HPP

#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Tags.hpp>
#include <impl/Kokkos_Reducer.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct FunctorPatternInterface {
  struct FOR {};
  struct REDUCE {};
  struct SCAN {};
};

/** \brief  Query Functor and execution policy argument tag for value type.
 *
 *  If C++11 enabled and 'value_type' is not explicitly declared then attempt
 *  to deduce the type from FunctorType::operator().
 */
template< typename PatternInterface , class Policy , class Functor >
struct FunctorAnalysis {
private:

  using FOR    = FunctorPatternInterface::FOR ;
  using REDUCE = FunctorPatternInterface::REDUCE ;
  using SCAN   = FunctorPatternInterface::SCAN ;

  //----------------------------------------

  template< typename P , bool = true >
  struct has_work_tag { using type = void ; };

  template< typename P >
  struct has_work_tag
    < P , ! std::is_same< typename P::work_tag , void >::value >
  {
    using type = typename P::work_tag ;
  };

  using Tag = typename has_work_tag< Policy >::type ;

  struct VOID {};

  using WTag = typename
    std::conditional< std::is_same<Tag,void>::value , VOID , Tag >::type ;

  //----------------------------------------
  // Check for Functor::value_type, which is either a simple type T or T[]

  template< typename F , bool = true >
  struct has_value_type { using type = void ; };

  template< typename F >
  struct has_value_type
    < F , ! std::is_same< typename F::value_type , void >::value >
  {
    using type = typename F::value_type ;

    static_assert( ! std::is_reference< type >::value &&
                   std::rank< type >::value <= 1 &&
                   std::extent< type >::value == 0
                 , "Kokkos Functor::value_type is T or T[]" );
  };

  //----------------------------------------
  // If Functor::value_type does not exist then evaluate operator(),
  // depending upon the pattern and whether the policy has a work tag,
  // to determine the reduction or scan value_type.

  template< typename F
          , typename P = PatternInterface
          , typename V = typename has_value_type<F>::type
          , bool     T = std::is_same< Tag , void >::value
          >
  struct deduce_value_type { using type = V ; };

  template< typename F >
  struct deduce_value_type< F , REDUCE , void , true > {

    template< typename M , typename A >
    KOKKOS_INLINE_FUNCTION static
    A deduce( void (Functor::*)( M , A & ) const );

    using type = decltype( deduce( & F::operator() ) );
  };

  template< typename F >
  struct deduce_value_type< F , REDUCE , void , false > {

    template< typename M , typename A >
    KOKKOS_INLINE_FUNCTION static
    A deduce( void (Functor::*)( WTag , M , A & ) const );

    template< typename M , typename A >
    KOKKOS_INLINE_FUNCTION static
    A deduce( void (Functor::*)( WTag const & , M , A & ) const );

    using type = decltype( deduce( & F::operator() ) );
  };

  template< typename F >
  struct deduce_value_type< F , SCAN , void , true > {

    template< typename M , typename A , typename I >
    KOKKOS_INLINE_FUNCTION static
    A deduce( void (Functor::*)( M , A & , I ) const );

    using type = decltype( deduce( & F::operator() ) );
  };

  template< typename F >
  struct deduce_value_type< F , SCAN , void , false > {

    template< typename M , typename A , typename I >
    KOKKOS_INLINE_FUNCTION static
    A deduce( void (Functor::*)( WTag , M , A & , I ) const );

    template< typename M , typename A , typename I >
    KOKKOS_INLINE_FUNCTION static
    A deduce( void (Functor::*)( WTag const & , M , A & , I ) const );

    using type = decltype( deduce( & F::operator() ) );
  };

  //----------------------------------------

  using candidate_type = typename deduce_value_type< Functor >::type ;

  enum { candidate_is_void  = std::is_same< candidate_type , void >::value
       , candidate_is_array = std::rank< candidate_type >::value == 1 };

  //----------------------------------------

public:

  using value_type = typename std::remove_extent< candidate_type >::type ;

  static_assert( ! std::is_const< value_type >::value
               , "Kokkos functor operator reduce argument cannot be const" );

private:

  // Stub to avoid defining a type 'void &'
  using ValueType = typename
    std::conditional< candidate_is_void , VOID , value_type >::type ;

public:

  using pointer_type = typename
    std::conditional< candidate_is_void , void , ValueType * >::type ;

  using reference_type = typename
    std::conditional< candidate_is_array  , ValueType * , typename
    std::conditional< ! candidate_is_void , ValueType & , void >
    ::type >::type ;

private:

  template< bool IsArray , class FF >
  KOKKOS_INLINE_FUNCTION static
  typename std::enable_if< IsArray , unsigned >::type
  get_length( FF const & f ) { return f.value_count ; }

  template< bool IsArray , class FF >
  KOKKOS_INLINE_FUNCTION static
  typename std::enable_if< ! IsArray , unsigned >::type
  get_length( FF const & ) { return 1 ; }

public:

  enum { StaticValueSize = ! candidate_is_void &&
                           ! candidate_is_array
                         ? sizeof(ValueType) : 0 };

  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_count( const Functor & f )
    { return FunctorAnalysis::template get_length< candidate_is_array >(f); }

  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_size( const Functor & f )
    { return FunctorAnalysis::template get_length< candidate_is_array >(f) * sizeof(ValueType); }

  //----------------------------------------

  template< class Unknown >
  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_count( const Unknown & )
    { return 1 ; }

  template< class Unknown >
  KOKKOS_FORCEINLINE_FUNCTION static
  unsigned value_size( const Unknown & )
    { return sizeof(ValueType); }

private:

  //----------------------------------------
  // parallel_reduce join operator

  template< class F , typename V , typename T , typename Enable = void >
  struct DeduceJoin
    {
      template< bool B , class FF >
      KOKKOS_INLINE_FUNCTION static
      typename std::enable_if< B , int >::type
      get_length( FF const & f ) { return f.value_count ; }

      template< bool B , class FF >
      KOKKOS_INLINE_FUNCTION static
      typename std::enable_if< ! B , int >::type
      get_length( FF const & ) { return 1 ; }

      KOKKOS_INLINE_FUNCTION
      static void join( F const & f
                      , V volatile * dst
                      , V volatile const * src
                      )
        {
          const int length = FunctorAnalysis::template get_length< candidate_is_array >( f );
          for ( int i = 0 ; i < length ; ++i ) dst[0] += src[0] ;
        }
    };

  template< class F , typename V , typename T , typename Enable = void >
  struct DeduceInit
    {
      template< bool B , class FF >
      KOKKOS_INLINE_FUNCTION static
      typename std::enable_if< B , int >::type
      get_length( FF const & f ) { return f.value_count ; }

      template< bool B , class FF >
      KOKKOS_INLINE_FUNCTION static
      typename std::enable_if< ! B , int >::type
      get_length( FF const & ) { return 1 ; }

      KOKKOS_INLINE_FUNCTION
      static void init( F const & f , V * dst )
        {
          const int length = FunctorAnalysis::template get_length< candidate_is_array >( f );
          for ( int i = 0 ; i < length ; ++i ) dst[0] = 0 ;
        }
    };

  // No tag and is array
  template< class F , typename V >
  struct DeduceJoin< F , V , void ,
    decltype( ((F*)0)->join( (V volatile *) 0
                           , (V volatile const *) 0 ) ) >
    {
      KOKKOS_INLINE_FUNCTION
      static void join( F const & f
                      , V volatile * dst
                      , V volatile const * src
                      )
        { f.join( dst , src ); };
    };

  // No tag and is reference
  template< class F , typename V >
  struct DeduceJoin< F , V , void ,
    decltype( ((F*)0)->join( *((V volatile *)0)
                           , *((V volatile const *)0) ) ) >
    {
      KOKKOS_INLINE_FUNCTION
      static void join( F const & f
                      , V volatile * dst
                      , V volatile const * src
                      )
        { f.join( *dst , *src ); }
    };

  // Tag and is array
  template< class F , typename V , typename T >
  struct DeduceJoin< F , V , T ,
    decltype( ((F*)0)->join( T()
                           , (V volatile *) 0
                           , (V volatile const *) 0 ) ) >
    {
      KOKKOS_INLINE_FUNCTION
      static void join( F const & f
                      , V volatile * dst
                      , V volatile const * src
                      )
        { f.join( T() , dst , src ); }
    };

  // Tag and is reference
  template< class F , typename V , typename T >
  struct DeduceJoin< F , V , T ,
    decltype( ((F*)0)->join( T()
                           , *((V volatile *)0)
                           , *((V volatile const *)0) ) ) >
    {
      KOKKOS_INLINE_FUNCTION
      static void join( F const & f
                      , V volatile * dst
                      , V volatile const * src
                      )
        { f.join( T() , *dst , *src ); }
    };

  // No tag and is array
  template< class F , typename V >
  struct DeduceInit< F , V , void ,
    decltype( ((F*)0)->init( (V*) 0 ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void init( F const & f , V * dst ) { f.init( dst ); };
    };

  // No tag and is reference
  template< class F , typename V >
  struct DeduceInit< F , V , void ,
    decltype( ((F*)0)->init( *((V*)0) ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void init( F const & f , V * dst ) { f.init( *dst ); }
    };

  // Tag and is array
  template< class F , typename V , typename T >
  struct DeduceInit< F , V , T ,
    decltype( ((F*)0)->init( T() , (V *) 0 ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void join( F const & f , V * dst ) { f.init( T() , dst ); }
    };

  // Tag and is reference
  template< class F , typename V , typename T >
  struct DeduceInit< F , T ,
    decltype( ((F*)0)->init( T() , *((V *)0) ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void join( F const & f , V * dst )
        { f.init( T() , *dst ); }
    };

  //----------------------------------------

public:

  struct Reducer
  {
  private:

    using Join = DeduceJoin< Functor, ValueType , Tag > ;
    using Init = DeduceInit< Functor, ValueType , Tag > ;

    Functor     const & m_functor ;
    ValueType * const   m_result ;
    int         const   m_length ;

  public:

    using reducer        = Reducer ;
    using value_type     = FunctorAnalysis::value_type ;
    using memory_space   = void ;
    using reference_type = FunctorAnalysis::reference_type ;

    KOKKOS_INLINE_FUNCTION
    void join( ValueType volatile * dst
             , ValueType volatile const * src ) const noexcept
      { Join::join( m_functor , dst , src ); }

    KOKKOS_INLINE_FUNCTION
    void init( ValueType * dst ) const noexcept
      { Init::init( m_functor , dst ); }

    KOKKOS_INLINE_FUNCTION explicit
    constexpr Reducer( Functor const & arg_functor
                     , ValueType     * arg_value = 0
                     , int             arg_length = 0 ) noexcept
      : m_functor( arg_functor ), m_result(arg_value), m_length(arg_length) {}

    KOKKOS_INLINE_FUNCTION
    constexpr int length() const noexcept { return m_length ; }

    KOKKOS_INLINE_FUNCTION
    ValueType & operator[]( int i ) const noexcept
      { return m_result[i]; }

  private:

    template< bool IsArray >
    constexpr
    typename std::enable_if< IsArray , ValueType * >::type
    ref() const noexcept { return m_result ; }

    template< bool IsArray >
    constexpr
    typename std::enable_if< ! IsArray , ValueType & >::type
    ref() const noexcept { return *m_result ; }

  public:

    KOKKOS_INLINE_FUNCTION
    auto result() const noexcept
      -> decltype( Reducer::template ref< candidate_is_array >() )
      { return Reducer::template ref< candidate_is_array >(); }
 };

  //----------------------------------------

private:

  template< class F , typename V , typename T , typename Enable = void >
  struct DeduceFinal
    {
      KOKKOS_INLINE_FUNCTION static
      void final( F const & , V * ) {}
    };

  // No tag and is array
  template< class F , typename V >
  struct DeduceFinal< F , V , void ,
    decltype( ((F*)0)->final( (V*) 0 ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void final( F const & f , V * dst ) { f.final( dst ); };
    };

  // No tag and is reference
  template< class F , typename V >
  struct DeduceFinal< F , V , void ,
    decltype( ((F*)0)->final( *((V*)0) ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void init( F const & f , V * dst ) { f.final( *dst ); }
    };

  // Tag and is array
  template< class F , typename V , typename T >
  struct DeduceFinal< F , V , T ,
    decltype( ((F*)0)->final( T() , (V *) 0 ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void join( F const & f , V * dst ) { f.final( T() , dst ); }
    };

  // Tag and is reference
  template< class F , typename V , typename T >
  struct DeduceFinal< F , T ,
    decltype( ((F*)0)->final( T() , *((V *)0) ) ) >
    {
      KOKKOS_INLINE_FUNCTION static
      void join( F const & f , V * dst )
        { f.final( T() , *dst ); }
    };

public:

  static void final( Functor const & f , ValueType * result )
    { DeduceFinal< Functor , ValueType , Tag >::final( f , result ); }

};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* KOKKOS_FUNCTORANALYSIS_HPP */

