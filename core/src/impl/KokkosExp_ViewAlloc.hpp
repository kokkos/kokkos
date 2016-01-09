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

  explicit
  ViewAllocateWithoutInitializing( const std::string & arg_label ) : label( arg_label ) {}

  explicit
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

template< unsigned N >
struct is_view_label< char[N] > : public std::true_type {};

template< unsigned N >
struct is_view_label< const char[N] > : public std::true_type {};

//----------------------------------------------------------------------------

template< typename ... P >
struct ViewAllocProp ;

/*  std::integral_constant<unsigned,I> are dummy arguments
 *  that avoid duplicate base class errors
 */
template< unsigned I >
struct ViewAllocProp< void , std::integral_constant<unsigned,I> >
{
  ViewAllocProp() = default ;
  ViewAllocProp( const ViewAllocProp & ) = default ;
  ViewAllocProp & operator = ( const ViewAllocProp & ) = default ;

  template< typename P >
  ViewAllocProp( const P & ) {}
};

/* Property flags have constexpr value */
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

/* Map input label type to std::string */
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

  typedef Kokkos::Impl::has_condition< void , Kokkos::Impl::is_memory_space , P ... >
    var_memory_space ;

  typedef Kokkos::Impl::has_condition< void , Kokkos::Impl::is_execution_space , P ... >
    var_execution_space ;

public:

  /* Flags for the common properties */
  enum { has_memory_space    = var_memory_space::value };
  enum { has_execution_space = var_execution_space::value };
  enum { has_label           = Kokkos::Impl::has_type< std::string , P... >::value };
  enum { allow_padding       = Kokkos::Impl::has_type< AllowPadding_t , P... >::value };
  enum { initialize          = ! Kokkos::Impl::has_type< WithoutInitializing_t , P ... >::value };

  typedef typename var_memory_space::type     memory_space ;
  typedef typename var_execution_space::type  execution_space ;

  /*  Copy from a matching argument list.
   *  Requires  std::is_same< P , ViewAllocProp< void , Args >::value ...
   */
  template< typename ... Args >
  ViewAllocProp( Args const & ... args )
    : ViewAllocProp< void , P >( args ) ...
    {}

  /* Copy from a matching property subset */
  template< typename ... Args >
  ViewAllocProp( ViewAllocProp< Args ... > const & arg )
    : ViewAllocProp< void , Args >( ((ViewAllocProp<void,Args> const &) arg ) ) ...
    {}
};

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif

