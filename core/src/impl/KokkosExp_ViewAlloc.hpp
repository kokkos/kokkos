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

#ifndef KOKKOS_EXPERIMENTAL_IMPL_VIEW_ALLOC_PROP_HPP
#define KOKKOS_EXPERIMENTAL_IMPL_VIEW_ALLOC_PROP_HPP

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#if defined( KOKKOS_USING_EXPERIMENTAL_VIEW )

namespace Kokkos {

/* For backward compatibility */

struct ViewAllocateWithoutInitializing {

  const std::string label ;

  ViewAllocateWithoutInitializing() : label() {}
  ViewAllocateWithoutInitializing( const std::string & arg_label ) : label( arg_label ) {}
  ViewAllocateWithoutInitializing( const char * const  arg_label ) : label( arg_label ) {}
};

} /* namespace Kokkos */

#endif

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

struct WithoutInitializing_t {};
struct AllowPadding_t {};
struct NullSpace_t {};

//----------------------------------------------------------------------------
/**\brief Whether a type can be used for a view label */

template < typename >
struct is_view_label : public std::false_type {};

template<>
struct is_view_label< std::string > : public std::true_type {};

template<>
struct is_view_label< char * > : public std::true_type {};

template<>
struct is_view_label< const char * > : public std::true_type {};

template< unsigned N >
struct is_view_label< char[N] > : public std::true_type {};

template< unsigned N >
struct is_view_label< const char[N] > : public std::true_type {};

//----------------------------------------------------------------------------


template< typename T , typename ... Args >
struct variadic_has_type ;

template< typename T >
struct variadic_has_type<T> { enum { value = false }; };

template< typename T , typename S , typename ... Args >
struct variadic_has_type<T,S,Args...>
{
private:
  enum { self_value = std::is_same<T,S>::value };
  enum { next_value = variadic_has_type<T,Args...>::value };

  static_assert( ! ( self_value && next_value )
               , "Variadic pack cannot have duplicated type" );

public:

  enum { value = self_value || next_value };
};

template< template<typename> class Condition , typename ... Args >
struct variadic_has_condition ;

template< template<typename> class Condition >
struct variadic_has_condition< Condition >
{
  enum { value = false };
  typedef void type ;
};

template< template<typename> class Condition , typename S , typename ... Args >
struct variadic_has_condition< Condition , S , Args... >
{
private:

  enum { self_value = Condition<S>::value };

  typedef variadic_has_condition< Condition , Args... > next ;

  static_assert( ! ( self_value && next::value )
               , "Variadic pack cannot have duplicated condition" );
public:

  enum { value = self_value || next::value };

  typedef typename
    std::conditional< self_value , S , typename next::type >::type
      type ;
};

//----------------------------------------------------------------------------

template< typename ... P >
struct ViewAllocProp ;

template<>
struct ViewAllocProp< void , void >
{
  ViewAllocProp() = default ;
  ViewAllocProp( const ViewAllocProp & ) = default ;
  ViewAllocProp & operator = ( const ViewAllocProp & ) = default ;

  template< typename P >
  ViewAllocProp( const P & ) {}
};

template< typename P >
struct ViewAllocProp
  < typename std::enable_if<
      std::is_same< P , AllowPadding_t >::value ||
      std::is_same< P , WithoutInitializing_t >::value
    >::type
  , P
  >
{
  ViewAllocProp() = default ;
  ViewAllocProp( const ViewAllocProp & ) = default ;
  ViewAllocProp & operator = ( const ViewAllocProp & ) = default ;

  typedef P type ;

  ViewAllocProp( const type & ) {}

  static constexpr type value = type();
};

template< typename Label >
struct ViewAllocProp
  < typename std::enable_if< is_view_label< Label >::value >::type
  , Label
  >
{
  ViewAllocProp() = default ;
  ViewAllocProp( const ViewAllocProp & ) = default ;
  ViewAllocProp & operator = ( const ViewAllocProp & ) = default ;

  typedef std::string type ;

  ViewAllocProp( const type & arg ) : value( arg ) {}
  ViewAllocProp( type && arg ) : value( arg ) {}

  type value ;
};

template< typename Space >
struct ViewAllocProp
  < typename std::enable_if<
      Kokkos::Impl::is_memory_space<Space>::value ||
      Kokkos::Impl::is_execution_space<Space>::value
    >::type
  , Space
  >
{
  ViewAllocProp() = default ;
  ViewAllocProp( const ViewAllocProp & ) = default ;
  ViewAllocProp & operator = ( const ViewAllocProp & ) = default ;

  typedef Space type ;

  ViewAllocProp( const type & arg ) : value( arg ) {}

  type value ;
};


template< typename ... P >
struct ViewAllocProp : public ViewAllocProp< void , P > ...
{
private:

  template< typename T >
  struct is_mem_space : public Kokkos::Impl::is_memory_space<T> {};

  template< typename T >
  struct is_exe_space : public Kokkos::Impl::is_execution_space<T> {};

  template< typename T >
  struct is_label : public Impl::is_view_label<T> {};

  typedef variadic_has_condition< is_mem_space , P ... > var_memory_space ;
  typedef variadic_has_condition< is_exe_space , P ... > var_execution_space ;
  typedef variadic_has_condition< is_label     , P ... > var_label ;

public:

  enum { has_label           = var_label::value };
  enum { has_memory_space    = var_memory_space::value };
  enum { has_execution_space = var_execution_space::value };
  enum { allow_padding       = variadic_has_type< AllowPadding_t , P... >::value };
  enum { initialize          = ! variadic_has_type< WithoutInitializing_t , P ... >::value };

  typedef typename var_memory_space::type     memory_space ;
  typedef typename var_execution_space::type  execution_space ;

  template< typename ... Args >
  ViewAllocProp( Args ... args )
    : ViewAllocProp< void , P >( args ) ...
    {}

  template< typename ... Args >
  ViewAllocProp( ViewAllocProp< Args ... > const & arg )
    : ViewAllocProp< void , P >( arg.template value<P>() ) ...
    {}

  template< typename T >
  static constexpr bool has_value()
    {
      return ! std::is_same<T,void>::value &&
             variadic_has_type<T,P...>::value ;
    }

  template< typename T >
  int value( typename std::enable_if<
               std::is_same<T,void>::value
           >::type * = 0 ) const
    { return 0 ; }
  
  template< typename T >
  const T & value( typename std::enable_if<
                     std::is_same<T,std::string>::value && has_label
                   >::type * = 0 ) const
    { return ViewAllocProp< void , typename var_label::type >::value ; }
  
  template< typename T >
  T value( typename std::enable_if<
                     std::is_same<T,std::string>::value && ! has_label
                   >::type * = 0 ) const
    { return T(); }

  template< typename T >
  const T & value( typename std::enable_if<
                     ! std::is_same<T,void>::value &&
                     ! std::is_same<T,std::string>::value &&
                     variadic_has_type<T,P...>::value
                   >::type * = 0 ) const
    { return ViewAllocProp<void,T>::value ; }

  template< typename T >
  T value( typename std::enable_if<
             ! std::is_same<T,void>::value &&
             ! std::is_same<T,std::string>::value &&
             ! variadic_has_type<T,P...>::value
           >::type * = 0 ) const
    { return T(); }
};

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif

