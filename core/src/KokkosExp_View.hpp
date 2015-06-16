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

#ifndef KOKKOS_EXPERIMENTAL_VIEW_HPP
#define KOKKOS_EXPERIMENTAL_VIEW_HPP

#include <string>
#include <type_traits>
#include <initializer_list>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

template< class > struct ViewDataAnalysis ;

template< class , class = void , typename Enable = void >
class ViewMapping { enum { is_assignable = false }; };

template< class DstMemorySpace , class SrcMemorySpace >
struct DeepCopy ;

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

/** \class ViewTraits
 *  \brief Traits class for accessing attributes of a View.
 *
 * This is an implementation detail of View.  It is only of interest
 * to developers implementing a new specialization of View.
 *
 * Template argument permutations:
 *   - View< DataType , void         , void         , void >
 *   - View< DataType , Space        , void         , void >
 *   - View< DataType , Space        , MemoryTraits , void >
 *   - View< DataType , Space        , void         , MemoryTraits >
 *   - View< DataType , ArrayLayout  , void         , void >
 *   - View< DataType , ArrayLayout  , Space        , void >
 *   - View< DataType , ArrayLayout  , MemoryTraits , void   >
 *   - View< DataType , ArrayLayout  , Space        , MemoryTraits >
 *   - View< DataType , MemoryTraits , void         , void  >
 */

template< class DataType ,
          class Arg1 = void ,
          class Arg2 = void ,
          class Arg3 = void >
class ViewTraits {
private:

  // Layout, Space, and MemoryTraits are optional
  // but need to appear in that order. That means Layout
  // can only be Arg1, Space can be Arg1 or Arg2, and
  // MemoryTraits can be Arg1, Arg2 or Arg3

  enum { Arg1IsLayout = Kokkos::Impl::is_array_layout<Arg1>::value };

  enum { Arg1IsSpace = Kokkos::Impl::is_space<Arg1>::value };
  enum { Arg2IsSpace = Kokkos::Impl::is_space<Arg2>::value };

  enum { Arg1IsMemoryTraits = Kokkos::Impl::is_memory_traits<Arg1>::value };
  enum { Arg2IsMemoryTraits = Kokkos::Impl::is_memory_traits<Arg2>::value };
  enum { Arg3IsMemoryTraits = Kokkos::Impl::is_memory_traits<Arg3>::value };

  enum { Arg1IsVoid = std::is_same< Arg1 , void >::value };
  enum { Arg2IsVoid = std::is_same< Arg2 , void >::value };
  enum { Arg3IsVoid = std::is_same< Arg3 , void >::value };

  static_assert( 1 == Arg1IsLayout + Arg1IsSpace + Arg1IsMemoryTraits + Arg1IsVoid
               , "Template argument #1 must be layout, space, traits, or void" );

  // If Arg1 is Layout       then Arg2 is Space, MemoryTraits, or void
  // If Arg1 is Space        then Arg2 is MemoryTraits or void
  // If Arg1 is MemoryTraits then Arg2 is void
  // If Arg1 is Void         then Arg2 is void

  static_assert( ( Arg1IsLayout       && ( 1 == Arg2IsSpace + Arg2IsMemoryTraits + Arg2IsVoid ) ) ||
                 ( Arg1IsSpace        && ( 0 == Arg2IsSpace ) && ( 1 == Arg2IsMemoryTraits + Arg2IsVoid ) ) ||
                 ( Arg1IsMemoryTraits && Arg2IsVoid ) ||
                 ( Arg1IsVoid         && Arg2IsVoid )
               , "Template argument #2 must be space, traits, or void" );

  // Arg3 is MemoryTraits or void and at most one argument is MemoryTraits
  static_assert( ( 1 == Arg3IsMemoryTraits + Arg3IsVoid ) &&
                 ( Arg1IsMemoryTraits + Arg2IsMemoryTraits + Arg3IsMemoryTraits <= 1 )
               , "Template argument #3 must be traits or void" );

  using ExecutionSpace =
    typename std::conditional< Arg1IsSpace , Arg1 ,
    typename std::conditional< Arg2IsSpace , Arg2 , Kokkos::DefaultExecutionSpace
    >::type >::type::execution_space ;

  using MemorySpace =
    typename std::conditional< Arg1IsSpace , Arg1 ,
    typename std::conditional< Arg2IsSpace , Arg2 , Kokkos::DefaultExecutionSpace
    >::type >::type::memory_space ;

  using HostMirrorSpace =
    typename Kokkos::Impl::is_space<
    typename std::conditional< Arg1IsSpace , Arg1 ,
    typename std::conditional< Arg2IsSpace , Arg2 , Kokkos::DefaultExecutionSpace
    >::type >::type >::host_mirror_space ;

  using ArrayLayout =
    typename std::conditional< Arg1IsLayout , Arg1 , typename ExecutionSpace::array_layout >::type ;

  // Arg1, Arg2, or Arg3 may be memory traits
  using MemoryTraits =
    typename std::conditional< Arg1IsMemoryTraits , Arg1 ,
    typename std::conditional< Arg2IsMemoryTraits , Arg2 ,
    typename std::conditional< Arg3IsMemoryTraits , Arg3 , MemoryManaged
    >::type >::type >::type ;

  using analysis = Kokkos::Experimental::Impl::ViewDataAnalysis< DataType > ;

public:

  //------------------------------------
  // Data type traits:

  using data_type           = typename analysis::type ;
  using const_data_type     = typename analysis::const_type ;
  using non_const_data_type = typename analysis::non_const_type ;

  //------------------------------------
  // Compatible array of trivial type traits:

  using array_scalar_type           = typename analysis::array_scalar_type ;
  using const_array_scalar_type     = typename analysis::const_array_scalar_type ;
  using non_const_array_scalar_type = typename analysis::non_const_array_scalar_type ;

  //------------------------------------
  // Value type traits:

  using value_type           = typename analysis::value_type ;
  using const_value_type     = typename analysis::const_value_type ;
  using non_const_value_type = typename analysis::non_const_value_type ;

  //------------------------------------
  // Mapping traits:

  using array_layout = ArrayLayout ;
  using dimension    = typename analysis::dimension ;
  using specialize   = typename analysis::specialize ; /* mapping specialization tag */

  enum { rank         = dimension::rank };
  enum { rank_dynamic = dimension::rank_dynamic };

  //------------------------------------
  // Execution space, memory space, memory access traits, and host mirror space.

  using execution_space   = ExecutionSpace ;
  using memory_space      = MemorySpace ;
  using device_type       = Device<ExecutionSpace,MemorySpace> ;
  using memory_traits     = MemoryTraits ;
  using host_mirror_space = HostMirrorSpace ;

  using size_type         = typename memory_space::size_type ;

  enum { is_hostspace      = std::is_same< memory_space , HostSpace >::value };
  enum { is_managed        = memory_traits::Unmanaged    == 0 };
  enum { is_random_access  = memory_traits::RandomAccess == 1 };

  //------------------------------------
};

/** \class View
 *  \brief View to an array of data.
 *
 * A View represents an array of one or more dimensions.
 * For details, please refer to Kokkos' tutorial materials.
 *
 * \section Kokkos_View_TemplateParameters Template parameters
 *
 * This class has both required and optional template parameters.  The
 * \c DataType parameter must always be provided, and must always be
 * first. The parameters \c Arg1Type, \c Arg2Type, and \c Arg3Type are
 * placeholders for different template parameters.  The default value
 * of the fifth template parameter \c Specialize suffices for most use
 * cases.  When explaining the template parameters, we won't refer to
 * \c Arg1Type, \c Arg2Type, and \c Arg3Type; instead, we will refer
 * to the valid categories of template parameters, in whatever order
 * they may occur.
 *
 * Valid ways in which template arguments may be specified:
 *   - View< DataType , Space >
 *   - View< DataType , Space  ,         MemoryTraits >
 *   - View< DataType , Space  , void  , MemoryTraits >
 *   - View< DataType , Layout , Space >
 *   - View< DataType , Layout , Space , MemoryTraits >
 *
 * \tparam DataType (required) This indicates both the type of each
 *   entry of the array, and the combination of compile-time and
 *   run-time array dimension(s).  For example, <tt>double*</tt>
 *   indicates a one-dimensional array of \c double with run-time
 *   dimension, and <tt>int*[3]</tt> a two-dimensional array of \c int
 *   with run-time first dimension and compile-time second dimension
 *   (of 3).  In general, the run-time dimensions (if any) must go
 *   first, followed by zero or more compile-time dimensions.  For
 *   more examples, please refer to the tutorial materials.
 *
 * \tparam Space (required) The memory space.
 *
 * \tparam Layout (optional) The array's layout in memory.  For
 *   example, LayoutLeft indicates a column-major (Fortran style)
 *   layout, and LayoutRight a row-major (C style) layout.  If not
 *   specified, this defaults to the preferred layout for the
 *   <tt>Space</tt>.
 *
 * \tparam MemoryTraits (optional) Assertion of the user's intended
 *   access behavior.  For example, RandomAccess indicates read-only
 *   access with limited spatial locality, and Unmanaged lets users
 *   wrap externally allocated memory in a View without automatic
 *   deallocation.
 *
 * \section Kokkos_View_MT MemoryTraits discussion
 *
 * \subsection Kokkos_View_MT_Interp MemoryTraits interpretation depends on Space
 *
 * Some \c MemoryTraits options may have different interpretations for
 * different \c Space types.  For example, with the Cuda device,
 * \c RandomAccess tells Kokkos to fetch the data through the texture
 * cache, whereas the non-GPU devices have no such hardware construct.
 *
 * \subsection Kokkos_View_MT_PrefUse Preferred use of MemoryTraits
 *
 * Users should defer applying the optional \c MemoryTraits parameter
 * until the point at which they actually plan to rely on it in a
 * computational kernel.  This minimizes the number of template
 * parameters exposed in their code, which reduces the cost of
 * compilation.  Users may always assign a View without specified
 * \c MemoryTraits to a compatible View with that specification.
 * For example:
 * \code
 * // Pass in the simplest types of View possible.
 * void
 * doSomething (View<double*, Cuda> out,
 *              View<const double*, Cuda> in)
 * {
 *   // Assign the "generic" View in to a RandomAccess View in_rr.
 *   // Note that RandomAccess View objects must have const data.
 *   View<const double*, Cuda, RandomAccess> in_rr = in;
 *   // ... do something with in_rr and out ...
 * }
 * \endcode
 */
template< class DataType
        , class Arg1 = void /* ArrayLayout, SpaceType, or MemoryTraits */
        , class Arg2 = void /* SpaceType or MemoryTraits */
        , class Arg3 = void /* MemoryTraits */ >
class View ;

} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

struct WithoutInitializing_t {};
struct AllowPadding_t {};

namespace {
constexpr WithoutInitializing_t WithoutInitializing = WithoutInitializing_t();
constexpr AllowPadding_t        AllowPadding        = AllowPadding_t();
}

template< class ... Parameters >
struct ViewAllocProp ;

template<>
struct ViewAllocProp<> {

  struct NullSpace {};

  using allow_padding_t = std::false_type ;
  using initialize_t    = std::true_type ;
  using memory_space    = NullSpace ;
  using execution_space = NullSpace ;

  const std::string      label ;
  const memory_space     memory ;
  const execution_space  execution ;
  const allow_padding_t  allow_padding ;
  const initialize_t     initialize ;

  ViewAllocProp()
    : label()
    , memory()
    , execution()
    , allow_padding()
    , initialize()
    {}

  ViewAllocProp( const ViewAllocProp & arg_prop )
    : label( arg_prop.label )
    , memory()
    , execution()
    , allow_padding()
    , initialize()
    {}

  ViewAllocProp( const std::string & arg_label , const ViewAllocProp & arg_prop = ViewAllocProp() )
    : label( arg_label )
    , memory()
    , execution()
    , allow_padding()
    , initialize()
    {}
};

template< class ... Parameters >
struct ViewAllocProp< WithoutInitializing_t , Parameters ... >
{
  using base_prop_type = ViewAllocProp< Parameters ... > ;

  using allow_padding_t = typename base_prop_type::allow_padding_t ;
  using initialize_t    = std::false_type ;
  using memory_space    = typename base_prop_type::memory_space ;
  using execution_space = typename base_prop_type::execution_space ;

  const std::string      label ;
  const memory_space     memory ;
  const execution_space  execution ;
  const allow_padding_t  allow_padding ;
  const initialize_t     initialize ;

  ViewAllocProp( const base_prop_type & arg_prop = base_prop_type() )
    : label( arg_prop.label  )
    , memory( arg_prop.memory )
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}

  ViewAllocProp( const std::string & arg_label , const ViewAllocProp & arg_prop )
    : label( arg_label )
    , memory( arg_prop.memory )
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}
};

template< class ... Parameters >
struct ViewAllocProp< AllowPadding_t , Parameters ... >
{
  using base_prop_type = ViewAllocProp< Parameters ... > ;

  using allow_padding_t = std::true_type ;
  using initialize_t    = typename base_prop_type::initialize_t ;
  using memory_space    = typename base_prop_type::memory_space ;
  using execution_space = typename base_prop_type::execution_space ;

  const std::string label ;
  const memory_space     memory ;
  const execution_space  execution ;
  const allow_padding_t  allow_padding ;
  const initialize_t     initialize ;

  ViewAllocProp( const base_prop_type & arg_prop = base_prop_type() )
    : label( arg_prop.label  )
    , memory( arg_prop.memory )
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}

  ViewAllocProp( const std::string & arg_label , const ViewAllocProp & arg_prop )
    : label( arg_label )
    , memory( arg_prop.memory )
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}
};

template< class Space , class ... Parameters >
struct ViewAllocProp< Space , Parameters ... >
{
  enum { is_exec = Kokkos::Impl::is_execution_space< Space >::value };
  enum { is_mem  = Kokkos::Impl::is_memory_space< Space >::value };

  static_assert( is_exec || is_mem , "View allocation given unknown parameter" );

  using base_prop_type = ViewAllocProp< Parameters ... > ;

  using allow_padding_t = typename base_prop_type::allow_padding_t ;
  using initialize_t    = typename base_prop_type::initialize_t ;
  using memory_space    = typename std::conditional< is_mem  , Space , typename base_prop_type::memory_space >::type ;
  using execution_space = typename std::conditional< is_exec , Space , typename base_prop_type::execution_space >::type ;

  const std::string      label ;
  const memory_space     memory ;
  const execution_space  execution ;
  const allow_padding_t  allow_padding ;
  const initialize_t     initialize ;

  template< class S >
  ViewAllocProp( const S & arg_space
               , typename std::enable_if
                   < std::is_same<S,memory_space>::value
                   , const base_prop_type &
                   >::type arg_prop = base_prop_type()
               )
    : label( arg_prop.label )
    , memory( arg_space )
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}

  template< class S >
  ViewAllocProp( const S & arg_space
               , typename std::enable_if
                   < std::is_same<S,execution_space>::value
                   , const base_prop_type &
                   >::type arg_prop = base_prop_type()
               )
    : label( arg_prop.label )
    , memory( arg_prop.memory )
    , execution( arg_space )
    , allow_padding()
    , initialize()
    {}

  ViewAllocProp( const std::string & arg_label , const ViewAllocProp & arg_prop )
    : label( arg_label )
    , memory( arg_prop.memory )
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}
};

template< class ExecSpace , class MemSpace >
struct ViewAllocProp< Kokkos::Device< ExecSpace , MemSpace > , std::string >
{
  using base_prop_type = ViewAllocProp<> ;

  using allow_padding_t = typename base_prop_type::allow_padding_t ;
  using initialize_t    = typename base_prop_type::initialize_t ;
  using memory_space    = MemSpace ;
  using execution_space = ExecSpace ;

  const std::string      label ;
  const memory_space     memory ;
  const execution_space  execution ;
  const allow_padding_t  allow_padding ;
  const initialize_t     initialize ;

  ViewAllocProp( const std::string & arg_label )
    : label( arg_label )
    , memory()
    , execution()
    , allow_padding()
    , initialize()
    {}
};

template< class ExecSpace , class MemSpace , unsigned N >
struct ViewAllocProp< Kokkos::Device< ExecSpace , MemSpace > , char[N] >
{
  using base_prop_type = ViewAllocProp<> ;

  using allow_padding_t = typename base_prop_type::allow_padding_t ;
  using initialize_t    = typename base_prop_type::initialize_t ;
  using memory_space    = MemSpace ;
  using execution_space = ExecSpace ;

  const std::string      label ;
  const memory_space     memory ;
  const execution_space  execution ;
  const allow_padding_t  allow_padding ;
  const initialize_t     initialize ;

  ViewAllocProp( const char * const arg_label )
    : label( arg_label )
    , memory()
    , execution()
    , allow_padding()
    , initialize()
    {}
};

template< class ExecSpace , class MemSpace , class ... Parameters >
struct ViewAllocProp< Kokkos::Device< ExecSpace , MemSpace >
                    , ViewAllocProp< Parameters ... >
                    >
{
  using base_prop_type = ViewAllocProp< Parameters ... > ;

  using allow_padding_t = typename base_prop_type::allow_padding_t ;
  using initialize_t    = typename base_prop_type::initialize_t ;
  using memory_space    = MemSpace ;

  using execution_space =
    typename std::conditional
      < Kokkos::Impl::is_execution_space< typename base_prop_type::execution_space >::value
      , typename base_prop_type::execution_space
      , ExecSpace
      >::type ;

  static_assert( std::is_same< typename base_prop_type::memory_space , ViewAllocProp<>::NullSpace >::value ||
                 std::is_same< typename base_prop_type::memory_space , memory_space >::value
               , "View allocation given incompatible memory space" );

  static_assert( Kokkos::Impl::VerifyExecutionCanAccessMemorySpace< typename execution_space::memory_space
                                                                  , memory_space >::value
               , "View allocation given incompatible execution space" );

  const std::string      label ;
  const memory_space     memory ;
  const execution_space  execution ;
  const allow_padding_t  allow_padding ;
  const initialize_t     initialize ;

  // If the input properties have a memory or execution space then copy construct those spaces
  // otherwise default construct those spaces.

  template< class P >
  ViewAllocProp( const P & arg_prop
               , typename std::enable_if
                   < std::is_same< P , base_prop_type >::value &&
                     Kokkos::Impl::is_memory_space< typename P::memory_space >::value &&
                     Kokkos::Impl::is_execution_space< typename P::memory_space >::value
                   >::type * = 0 )
    : label( arg_prop.label )
    , memory( arg_prop.memory )
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}

  template< class P >
  ViewAllocProp( const P & arg_prop
               , typename std::enable_if
                   < std::is_same< P , base_prop_type >::value &&
                     Kokkos::Impl::is_memory_space< typename P::memory_space >::value &&
                     ! Kokkos::Impl::is_execution_space< typename P::execution_space >::value
                   >::type * = 0 )
    : label( arg_prop.label )
    , memory( arg_prop.memory )
    , execution()
    , allow_padding()
    , initialize()
    {}

  template< class P >
  ViewAllocProp( const P & arg_prop
               , typename std::enable_if
                   < std::is_same< P , base_prop_type >::value &&
                     ! Kokkos::Impl::is_memory_space< typename P::memory_space >::value &&
                     Kokkos::Impl::is_execution_space< typename P::execution_space >::value
                   >::type * = 0 )
    : label( arg_prop.label )
    , memory()
    , execution( arg_prop.execution )
    , allow_padding()
    , initialize()
    {}

  template< class P >
  ViewAllocProp( const P & arg_prop
               , typename std::enable_if
                   < std::is_same< P , base_prop_type >::value &&
                     ! Kokkos::Impl::is_memory_space< typename P::memory_space >::value &&
                     ! Kokkos::Impl::is_execution_space< typename P::execution_space >::value
                   >::type * = 0 )
    : label( arg_prop.label )
    , memory()
    , execution()
    , allow_padding()
    , initialize()
    {}
};




ViewAllocProp<>
inline
view_alloc(void)
{ return ViewAllocProp<>(); }

template< class ... Args >
inline
ViewAllocProp< Args ... >
view_alloc( const std::string & arg_label , Args ... args )
{
  return ViewAllocProp< Args ... >( arg_label , view_alloc( args ... ) );
}

template< class ... Args >
inline
ViewAllocProp< Args ... >
view_alloc( const char * const arg_label , Args ... args )
{
  return ViewAllocProp< Args ... >( arg_label , view_alloc( args ... ) );
}

template< class ... Args >
inline
ViewAllocProp< WithoutInitializing_t , Args ... >
view_alloc( const WithoutInitializing_t & , Args ... args )
{
  return ViewAllocProp< WithoutInitializing_t, Args ... >( view_alloc( args ... ) );
}

template< class ... Args >
inline
ViewAllocProp< AllowPadding_t , Args ... >
view_alloc( const AllowPadding_t & , Args ... args )
{
  return ViewAllocProp< AllowPadding_t, Args ... >( view_alloc( args ... ) );
}

template< class Space , class ... Args >
inline
typename std::enable_if
  < Kokkos::Impl::is_memory_space< Space >::value
  , ViewAllocProp< Space , Args ... >
  >::type
view_alloc( const Space & space , Args ... args )
{
  return ViewAllocProp< Space , Args ... >( space , view_alloc( args ... ) );
};

template< class Space , class ... Args >
inline
typename std::enable_if
  < Kokkos::Impl::is_execution_space< Space >::value
  , ViewAllocProp< Space , Args ... >
  >::type
view_alloc( const Space & space , Args ... args )
{
  return ViewAllocProp< Space , Args ... >( space , view_alloc( args ... ) );
};

} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <impl/KokkosExp_ViewMapping.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

/**\brief  Each R? template argument designates whether the subview argument is a range */
template< class V
        , bool R0 = false , bool R1 = false , bool R2 = false , bool R3 = false
        , bool R4 = false , bool R5 = false , bool R6 = false , bool R7 = false >
using Subview = typename Kokkos::Experimental::Impl::SubviewType< V, R0 , R1 , R2 , R3 , R4 , R5 , R6 , R7 >::type ;

template< class DataType , class Arg1 , class Arg2 , class Arg3 >
class View : public ViewTraits< DataType , Arg1 , Arg2 , Arg3 > {
private:

  template< class , class , class , class > friend class View ;

  using traits       = ViewTraits< DataType , Arg1 , Arg2 , Arg3 > ;
  using map_type     = Kokkos::Experimental::Impl::ViewMapping< traits > ;
  using track_type   = Kokkos::Experimental::Impl::SharedAllocationTracker ;

  track_type  m_track ;
  map_type    m_map ;

public:

  //----------------------------------------
  /** \brief  Compatible view of array of scalar types */
  using array_type = View< typename traits::array_scalar_type ,
                           typename traits::array_layout ,
                           typename traits::device_type ,
                           typename traits::memory_traits > ;

  /** \brief  Compatible view of const data type */
  using const_type = View< typename traits::const_data_type ,
                           typename traits::array_layout ,
                           typename traits::device_type ,
                           typename traits::memory_traits > ;

  /** \brief  Compatible view of non-const data type */
  using non_const_type = View< typename traits::non_const_data_type ,
                               typename traits::array_layout ,
                               typename traits::device_type ,
                               typename traits::memory_traits > ;

  /** \brief  Compatible HostMirror view */
  using HostMirror = View< typename traits::non_const_data_type ,
                           typename traits::array_layout ,
                           typename traits::host_mirror_space ,
                           void > ;

  //----------------------------------------
  // Domain dimensions

  enum { Rank = map_type::Rank };

  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0() { return m_map.dimension_0(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_1() { return m_map.dimension_1(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_2() { return m_map.dimension_2(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_3() { return m_map.dimension_3(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_4() { return m_map.dimension_4(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_5() { return m_map.dimension_5(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_6() { return m_map.dimension_6(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_7() { return m_map.dimension_7(); }

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() { return m_map.stride_0(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() { return m_map.stride_1(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() { return m_map.stride_2(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() { return m_map.stride_3(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() { return m_map.stride_4(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() { return m_map.stride_5(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() { return m_map.stride_6(); }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() { return m_map.stride_7(); }

  //----------------------------------------
  // Range extent

  using reference_type = typename map_type::reference_type ;

  enum { reference_type_is_lvalue_reference = std::is_lvalue_reference< reference_type >::value };

  KOKKOS_INLINE_FUNCTION constexpr size_t extent() { return m_map.extent(); }
  KOKKOS_INLINE_FUNCTION constexpr bool   extent_is_contiguous() { return m_map.extent_is_contiguous(); }
  KOKKOS_INLINE_FUNCTION constexpr typename traits::value_type * data() { return m_map.data(); }

  //----------------------------------------

private:

  using scalar_operator_reference_type =
    typename std::conditional< Rank == 0 , reference_type
      , Kokkos::Experimental::Impl::Error_view_scalar_reference_to_non_scalar_view >::type ;

  using scalar_operator_index_type =
    typename std::conditional< Rank == 0 , const int
      , Kokkos::Experimental::Impl::Error_view_scalar_reference_to_non_scalar_view >::type ;

public:

  // Rank == 0

  KOKKOS_FORCEINLINE_FUNCTION
  scalar_operator_reference_type operator()() const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, 0, 0, 0, 0, 0, 0, 0, 0 );
      return scalar_operator_reference_type( m_map.reference() );
    }

  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( scalar_operator_index_type i0
            , const int i1 = 0 , const int i2 = 0 , const int i3 = 0
            , const int i4 = 0 , const int i5 = 0 , const int i6 = 0 , const int i7 = 0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference();
    }

  // Rank == 1

  template< typename I0 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 1 && std::is_integral<I0>::value
                          ), reference_type >::type
  operator[]( const I0 & i0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, 0, 0, 0, 0, 0, 0, 0 );
      return m_map.reference(i0);
    }

  template< typename I0 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 1 && std::is_integral<I0>::value
                          ), reference_type >::type
  operator()( const I0 & i0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, 0, 0, 0, 0, 0, 0, 0 );
      return m_map.reference(i0);
    }

  template< typename I0 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( const I0 & i0
            , typename std::enable_if<( Rank == 1 && std::is_integral<I0>::value ), const int >::type i1
            , const int i2 = 0 , const int i3 = 0
            , const int i4 = 0 , const int i5 = 0 , const int i6 = 0 , const int i7 = 0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0);
    }

  // Rank == 2

  template< typename I0 , typename I1 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 2 &&
                            std::is_integral<I0>::value &&
                            std::is_integral<I1>::value
                          ), reference_type >::type
  operator()( const I0 & i0 , const I1 & i1 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, 0, 0, 0, 0, 0, 0 );
      return m_map.reference(i0,i1);
    }

  template< typename I0 , typename I1 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( const I0 & i0 , const I1 & i1
            , typename std::enable_if<( Rank == 2 &&
                                        std::is_integral<I0>::value &&
                                        std::is_integral<I1>::value
                                      ), const int >::type i2
            , const int i3 = 0
            , const int i4 = 0 , const int i5 = 0 , const int i6 = 0 , const int i7 = 0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0,i1);
    }

  // Rank == 3

  template< typename I0 , typename I1 , typename I2 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 3 &&
                            std::is_integral<I0>::value &&
                            std::is_integral<I1>::value &&
                            std::is_integral<I2>::value 
                          ), reference_type >::type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, 0, 0, 0, 0, 0 );
      return m_map.reference(i0,i1,i2);
    }

  template< typename I0 , typename I1 , typename I2 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2
            , typename std::enable_if<( Rank == 3 &&
                                        std::is_integral<I0>::value &&
                                        std::is_integral<I1>::value &&
                                        std::is_integral<I2>::value
                                      ), const int >::type i3
            , const int i4 = 0 , const int i5 = 0 , const int i6 = 0 , const int i7 = 0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0,i1,i2);
    }

  // Rank == 4

  template< typename I0 , typename I1 , typename I2 , typename I3 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 4 &&
                            std::is_integral<I0>::value &&
                            std::is_integral<I1>::value &&
                            std::is_integral<I2>::value &&
                            std::is_integral<I3>::value
                          ), reference_type >::type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, 0, 0, 0, 0 );
      return m_map.reference(i0,i1,i2,i3);
    }

  template< typename I0 , typename I1 , typename I2 , typename I3 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , typename std::enable_if<( Rank == 4 &&
                                        std::is_integral<I0>::value &&
                                        std::is_integral<I1>::value &&
                                        std::is_integral<I2>::value &&
                                        std::is_integral<I3>::value
                                      ), const int >::type i4
            , const int i5 = 0 , const int i6 = 0 , const int i7 = 0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0,i1,i2,i3);
    }

  // Rank == 5

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 5 &&
                            std::is_integral<I0>::value &&
                            std::is_integral<I1>::value &&
                            std::is_integral<I2>::value &&
                            std::is_integral<I3>::value &&
                            std::is_integral<I4>::value 
                          ), reference_type >::type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , const I4 & i4 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, 0, 0, 0 );
      return m_map.reference(i0,i1,i2,i3,i4);
    }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , const I4 & i4
            , typename std::enable_if<( Rank == 5 &&
                                        std::is_integral<I0>::value &&
                                        std::is_integral<I1>::value &&
                                        std::is_integral<I2>::value &&
                                        std::is_integral<I3>::value &&
                                        std::is_integral<I4>::value
                                      ), const int >::type i5
            , const int i6 = 0 , const int i7 = 0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0,i1,i2,i3,i4);
    }

  // Rank == 6

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 6 &&
                            std::is_integral<I0>::value &&
                            std::is_integral<I1>::value &&
                            std::is_integral<I2>::value &&
                            std::is_integral<I3>::value &&
                            std::is_integral<I4>::value &&
                            std::is_integral<I5>::value
                          ), reference_type >::type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , const I4 & i4 , const I5 & i5 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, 0, 0 );
      return m_map.reference(i0,i1,i2,i3,i4,i5);
    }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , const I4 & i4 , const I5 & i5
            , typename std::enable_if<( Rank == 6 &&
                                        std::is_integral<I0>::value &&
                                        std::is_integral<I1>::value &&
                                        std::is_integral<I2>::value &&
                                        std::is_integral<I3>::value &&
                                        std::is_integral<I4>::value
                                      ), const int >::type i6
            , const int i7 = 0 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0,i1,i2,i3,i4,i5);
    }

  // Rank == 7

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 , typename I6 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 7 &&
                            std::is_integral<I0>::value &&
                            std::is_integral<I1>::value &&
                            std::is_integral<I2>::value &&
                            std::is_integral<I3>::value &&
                            std::is_integral<I4>::value &&
                            std::is_integral<I5>::value &&
                            std::is_integral<I6>::value 
                          ), reference_type >::type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , const I4 & i4 , const I5 & i5 , const I6 & i6 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, 0 );
      return m_map.reference(i0,i1,i2,i3,i4,i5,i6);
    }

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 , typename I6 >
  KOKKOS_FORCEINLINE_FUNCTION
  reference_type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , const I4 & i4 , const I5 & i5 , const I6 & i6
            , typename std::enable_if<( Rank == 7 &&
                                        std::is_integral<I0>::value &&
                                        std::is_integral<I1>::value &&
                                        std::is_integral<I2>::value &&
                                        std::is_integral<I3>::value &&
                                        std::is_integral<I4>::value
                                      ), const int >::type i7
            ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0,i1,i2,i3,i4,i5,i6);
    }

  // Rank == 8

  template< typename I0 , typename I1 , typename I2 , typename I3
          , typename I4 , typename I5 , typename I6 , typename I7 >
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<( Rank == 8 &&
                            std::is_integral<I0>::value &&
                            std::is_integral<I1>::value &&
                            std::is_integral<I2>::value &&
                            std::is_integral<I3>::value &&
                            std::is_integral<I4>::value &&
                            std::is_integral<I5>::value &&
                            std::is_integral<I6>::value &&
                            std::is_integral<I7>::value
                          ), reference_type >::type
  operator()( const I0 & i0 , const I1 & i1 , const I2 & i2 , const I3 & i3
            , const I4 & i4 , const I5 & i5 , const I6 & i6 , const I7 & i7 ) const
    {
      KOKKOS_ASSERT_VIEW_MAPPING_ACCESS( typename traits::memory_space, m_map, Rank, i0, i1, i2, i3, i4, i5, i6, i7 );
      return m_map.reference(i0,i1,i2,i3,i4,i5,i6,i7);
    }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  ~View() {}

  KOKKOS_INLINE_FUNCTION
  View() : m_track(), m_map() {}

  KOKKOS_INLINE_FUNCTION
  View( const View & rhs ) : m_track( rhs.m_track ), m_map( rhs.m_map ) {}

  KOKKOS_INLINE_FUNCTION
  View( View && rhs ) : m_track( rhs.m_track ), m_map( rhs.m_map ) {}

  KOKKOS_INLINE_FUNCTION
  View & operator = ( const View & rhs ) { m_track = rhs.m_track ; m_map = rhs.m_map ; return *this ; }

  KOKKOS_INLINE_FUNCTION
  View & operator = ( View && rhs ) { m_track = rhs.m_track ; m_map = rhs.m_map ; return *this ; }

  //----------------------------------------

  template< class RT , class R1 , class R2 , class R3 >
  KOKKOS_INLINE_FUNCTION
  View( const View<RT,R1,R2,R3> & rhs )
    : m_track( rhs.m_track )
    , m_map()
    {
      using SrcTraits = typename View<RT,R1,R2,R3>::traits ;
      using Mapping   = Kokkos::Experimental::Impl::ViewMapping< traits , SrcTraits > ;
      static_assert( Mapping::is_assignable , "Incompatible View copy construction" );
      Mapping::assign( m_map , rhs.m_map , rhs.m_track );
    }

  template< class RT , class R1 , class R2 , class R3 >
  KOKKOS_INLINE_FUNCTION
  View( View<RT,R1,R2,R3> && rhs )
    : m_track( rhs.m_track )
    , m_map()
    {
      using SrcTraits = typename View<RT,R1,R2,R3>::traits ;
      using Mapping   = Kokkos::Experimental::Impl::ViewMapping< traits , SrcTraits > ;
      static_assert( Mapping::is_assignable , "Incompatible View move construction" );
      Mapping::assign( m_map , rhs.m_map , rhs.m_track );
    }

  template< class RT , class R1 , class R2 , class R3 >
  KOKKOS_INLINE_FUNCTION
  View & operator = ( const View<RT,R1,R2,R3> & rhs )
    {
      using SrcTraits = typename View<RT,R1,R2,R3>::traits ;
      using Mapping   = Kokkos::Experimental::Impl::ViewMapping< traits , SrcTraits > ;
      static_assert( Mapping::is_assignable , "Incompatible View copy assignment" );
      Mapping::assign( m_map , rhs.m_map , rhs.m_track );
      m_track.operator=( rhs.m_track );
      return *this ;
    }

  template< class RT , class R1 , class R2 , class R3 >
  KOKKOS_INLINE_FUNCTION
  View & operator = ( View<RT,R1,R2,R3> && rhs )
    {
      using SrcTraits = typename View<RT,R1,R2,R3>::traits ;
      using Mapping   = Kokkos::Experimental::Impl::ViewMapping< traits , SrcTraits > ;
      static_assert( Mapping::is_assignable , "Incompatible View move assignment" );
      Mapping::assign( m_map , rhs.m_map , rhs.m_track );
      m_track.operator=( rhs.m_track );
      return *this ;
    }

  //----------------------------------------
  // Allocation according to allocation properties

private:

  // Must call destructor for non-trivial types
  template< class ExecSpace >
  struct DestroyFunctor {
    map_type  m_map ;
    ExecSpace m_space ;

    KOKKOS_INLINE_FUNCTION
    void destroy_shared_allocation() { m_map.destroy( m_space ); }
  };

public:

  inline
  const std::string label() const { return m_track.template get_label< typename traits::memory_space >(); }

  template< class Prop >
  explicit inline
  View( const Prop & arg_prop
      , const size_t arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0
      )
    : m_track()
    , m_map()
    {
      // Merge the < execution_space , memory_space > into the properties.
      using alloc_prop = ViewAllocProp< typename traits::device_type , Prop > ;

      using execution_space = typename alloc_prop::execution_space ;
      using memory_space    = typename traits::memory_space ;
      using destroy_functor = DestroyFunctor< execution_space > ;
      using record_type     = Kokkos::Experimental::Impl::SharedAllocationRecord< memory_space , destroy_functor > ;

      static_assert( traits::is_managed , "View allocation constructor requires managed memory" );

      const alloc_prop prop( arg_prop );

      // Query the mapping for byte-size of allocation.
      const size_t alloc_size = map_type::memory_extent( prop.allow_padding
                                                       , arg_N0 , arg_N1 , arg_N2 , arg_N3
                                                       , arg_N4 , arg_N5 , arg_N6 , arg_N7 );

      // Allocate memory from the memory space.
      record_type * const record = record_type::allocate( prop.memory , prop.label , alloc_size );

      // Construct the mapping object prior to start of tracking
      // to assign destroy functor and possibly initialize.
      m_map = map_type( record->data()
                      , prop.allow_padding
                      , arg_N0 , arg_N1 , arg_N2 , arg_N3
                      , arg_N4 , arg_N5 , arg_N6 , arg_N7 );

      // Copy the destroy functor into the allocation record before initiating tracking.
      record->m_destroy.m_map   = m_map ;
      record->m_destroy.m_space = prop.execution ;

      if ( prop.initialize.value ) {
        m_map.construct( prop.execution );
      }

      // Destory functor assigned and initialization complete, start tracking
      m_track = track_type( record );
    }

  //----------------------------------------

  // Memory extent required to wrap these dimensions.
  constexpr static size_t memory_extent( const size_t arg_N0 = 0
                                       , const size_t arg_N1 = 0
                                       , const size_t arg_N2 = 0
                                       , const size_t arg_N3 = 0
                                       , const size_t arg_N4 = 0
                                       , const size_t arg_N5 = 0
                                       , const size_t arg_N6 = 0
                                       , const size_t arg_N7 = 0
                                       )
    {
      return map_type::memory_extent( std::integral_constant<bool,false>()
                                    , arg_N0 , arg_N1 , arg_N2 , arg_N3
                                    , arg_N4 , arg_N5 , arg_N6 , arg_N7 );
    }

  explicit inline
  View( typename traits::value_type * arg_ptr
      , const size_t arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0
      )
    : m_track() // No memory tracking
    , m_map( arg_ptr , std::integral_constant<bool,false>()
           , arg_N0 , arg_N1 , arg_N2 , arg_N3
           , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    {}

  //----------------------------------------
  // Shared scratch memory constructor

  static inline
  unsigned shmem_size( const unsigned arg_N0 = 0 ,
                       const unsigned arg_N1 = 0 ,
                       const unsigned arg_N2 = 0 ,
                       const unsigned arg_N3 = 0 ,
                       const unsigned arg_N4 = 0 ,
                       const unsigned arg_N5 = 0 ,
                       const unsigned arg_N6 = 0 ,
                       const unsigned arg_N7 = 0 )
  {
    return map_type::memory_extent( std::integral_constant<bool,false>()
                                  , arg_N0 , arg_N1 , arg_N2 , arg_N3
                                  , arg_N4 , arg_N5 , arg_N6 , arg_N7 );
  }

  explicit KOKKOS_INLINE_FUNCTION
  View( const typename traits::execution_space::scratch_memory_space & arg_space
      , const unsigned arg_N0 = 0 
      , const unsigned arg_N1 = 0
      , const unsigned arg_N2 = 0
      , const unsigned arg_N3 = 0
      , const unsigned arg_N4 = 0
      , const unsigned arg_N5 = 0
      , const unsigned arg_N6 = 0
      , const unsigned arg_N7 = 0 )
    : m_track() // No memory tracking
    , m_map( arg_space.get_shmem( map_type::memory_extent( std::integral_constant<bool,false>()
                                                         , arg_N0 , arg_N1 , arg_N2 , arg_N3
                                                         , arg_N4 , arg_N5 , arg_N6 , arg_N7 ) )
           , std::integral_constant<bool,false>() 
           , arg_N0 , arg_N1 , arg_N2 , arg_N3
           , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
    {}

  //----------------------------------------
  // Subviews

private:

  explicit KOKKOS_INLINE_FUNCTION
  View( const track_type & rhs )
    : m_track( rhs )
    , m_map()
    {}

public:

  template< class D , class A1 , class A2 , class A3
          , class T0 , class T1 , class T2 , class T3
          , class T4 , class T5 , class T6 , class T7 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D , A1 , A2 , A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T6>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T7>::is_range
    >
  subview( const View< D , A1 , A2 , A3 > & src
         , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
         , T4 const & arg4 , T5 const & arg5 , T6 const & arg6 , T7 const & arg7
         );
  
  template< class D , class A1 , class A2 , class A3
          , class T0 , class T1 , class T2 , class T3
          , class T4 , class T5 , class T6 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D , A1 , A2 , A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T6>::is_range
    >
  subview( const View< D , A1 , A2 , A3 > & src
         , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
         , T4 const & arg4 , T5 const & arg5 , T6 const & arg6
         );

  template< class D , class A1 , class A2 , class A3
          , class T0 , class T1 , class T2 , class T3
          , class T4 , class T5 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D , A1 , A2 , A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
    >
  subview( const View< D , A1 , A2 , A3 > & src
         , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
         , T4 const & arg4 , T5 const & arg5
         );
  
  template< class D , class A1 , class A2 , class A3
          , class T0 , class T1 , class T2 , class T3
          , class T4 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D , A1 , A2 , A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    >
  subview( const View< D , A1 , A2 , A3 > & src
         , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
         , T4 const & arg4
         );
  
  template< class D , class A1 , class A2 , class A3
          , class T0 , class T1 , class T2 , class T3 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D , A1 , A2 , A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    >
  subview( const View< D , A1 , A2 , A3 > & src
         , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
         );
  
  template< class D , class A1 , class A2 , class A3
          , class T0 , class T1 , class T2 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D , A1 , A2 , A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    >
  subview( const View< D , A1 , A2 , A3 > & src
         , T0 const & arg0 , T1 const & arg1 , T2 const & arg2
         );

  template< class D , class A1 , class A2 , class A3
          , class T0 , class T1 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D , A1 , A2 , A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    >
  subview( const View< D , A1 , A2 , A3 > & src
         , T0 const & arg0 , T1 const & arg1
         );
  
  template< class D, class A1, class A2, class A3, class T0 >
  friend
  KOKKOS_INLINE_FUNCTION
  Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    >
  subview( const View< D, A1, A2, A3 > & src , T0 const & arg0 );

};

template< class > struct is_view : public std::false_type {};

template< class D, class A1, class A2, class A3 >
struct is_view< View<D,A1,A2,A3> > : public std::true_type {};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template< class D, class A1, class A2, class A3
        , class T0 , class T1 , class T2 , class T3
        , class T4 , class T5 , class T6 , class T7 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T6>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T7>::is_range
  >
subview( const View< D, A1, A2, A3 > & src
       , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
       , T4 const & arg4 , T5 const & arg5 , T6 const & arg6 , T7 const & arg7
       )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T6>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T7>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 8 , "Subview of rank 8 View requires 8 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map, src.m_map, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 );

  return dst ;
}

template< class D, class A1, class A2, class A3
        , class T0 , class T1 , class T2 , class T3
        , class T4 , class T5 , class T6 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T6>::is_range
  >
subview( const View< D, A1, A2, A3 > & src
       , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
       , T4 const & arg4 , T5 const & arg5 , T6 const & arg6
       )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T6>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 7 , "Subview of rank 7 View requires 7 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map, src.m_map, arg0, arg1, arg2, arg3, arg4, arg5, arg6, 0 );

  return dst ;
}

template< class D, class A1, class A2, class A3
        , class T0 , class T1 , class T2 , class T3
        , class T4 , class T5 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
  >
subview( const View< D, A1, A2, A3 > & src
       , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
       , T4 const & arg4 , T5 const & arg5
       )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T5>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 6 , "Subview of rank 6 View requires 6 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map, src.m_map, arg0, arg1, arg2, arg3, arg4, arg5, 0, 0 );

  return dst ;
}

template< class D, class A1, class A2, class A3
        , class T0 , class T1 , class T2 , class T3
        , class T4 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
  >
subview( const View< D, A1, A2, A3 > & src
       , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
       , T4 const & arg4
       )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T4>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 5 , "Subview of rank 5 View requires 5 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map, src.m_map, arg0, arg1, arg2, arg3, arg4, 0, 0, 0 );

  return dst ;
}

template< class D, class A1, class A2, class A3
        , class T0 , class T1 , class T2 , class T3 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
  >
subview( const View< D, A1, A2, A3 > & src
       , T0 const & arg0 , T1 const & arg1 , T2 const & arg2 , T3 const & arg3
       )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T3>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 4 , "Subview of rank 4 View requires 4 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map, src.m_map, arg0, arg1, arg2, arg3, 0, 0, 0, 0 );

  return dst ;
}

template< class D, class A1, class A2, class A3
        , class T0 , class T1 , class T2 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
  >
subview( const View< D, A1, A2, A3 > & src
       , T0 const & arg0 , T1 const & arg1 , T2 const & arg2
       )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T2>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 3 , "Subview of rank 3 View requires 3 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map, src.m_map, arg0, arg1, arg2, 0, 0, 0, 0, 0 );

  return dst ;
}

template< class D, class A1, class A2, class A3
        , class T0 , class T1 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
  >
subview( const View< D, A1, A2, A3 > & src
       , T0 const & arg0 , T1 const & arg1
       )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    , Kokkos::Experimental::Impl::ViewOffsetRange<T1>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 2 , "Subview of rank 2 View requires 2 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map, src.m_map, arg0, arg1, 0, 0, 0, 0, 0, 0 );

  return dst ;
}

template< class D, class A1, class A2, class A3, class T0 >
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::Subview< View< D, A1, A2, A3 >
  , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
  >
subview( const View< D, A1, A2, A3 > & src , T0 const & arg0 )
{
  using SrcView = View< D, A1, A2, A3 > ;

  using Mapping = Kokkos::Experimental::Impl::SubviewMapping
    < typename SrcView::traits
    , Kokkos::Experimental::Impl::ViewOffsetRange<T0>::is_range
    > ;

  using DstView = typename Mapping::type ;

  static_assert( SrcView::Rank == 1 , "Subview of rank 1 View requires 1 arguments" );

  DstView dst( src.m_track );

  Mapping::assign( dst.m_map , src.m_map , arg0, 0, 0, 0, 0, 0, 0, 0 );

  return dst ;
}

} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

template< class LT , class L1 , class L2 , class L3
        , class RT , class R1 , class R2 , class R3 >
KOKKOS_INLINE_FUNCTION
bool operator == ( const View<LT,L1,L2,L3> & lhs ,
                   const View<RT,R1,R2,R3> & rhs )
{
  // Same data, layout, dimensions
  using lhs_traits = ViewTraits<LT,L1,L2,L3> ;
  using rhs_traits = ViewTraits<RT,R1,R2,R3> ;

  return
    std::is_same< typename lhs_traits::const_value_type ,
                  typename rhs_traits::const_value_type >::value &&
    std::is_same< typename lhs_traits::array_layout ,
                  typename rhs_traits::array_layout >::value &&
    std::is_same< typename lhs_traits::memory_space ,
                  typename rhs_traits::memory_space >::value &&
    lhs_traits::Rank == rhs_traits::Rank &&
    lhs.data()        == rhs.data() &&
    lhs.extent()      == rhs.extent() &&
    lhs.dimension_0() == rhs.dimension_0() &&
    lhs.dimension_1() == rhs.dimension_1() &&
    lhs.dimension_2() == rhs.dimension_2() &&
    lhs.dimension_3() == rhs.dimension_3() &&
    lhs.dimension_4() == rhs.dimension_4() &&
    lhs.dimension_5() == rhs.dimension_5() &&
    lhs.dimension_6() == rhs.dimension_6() &&
    lhs.dimension_7() == rhs.dimension_7();
}

template< class LT , class L1 , class L2 , class L3
        , class RT , class R1 , class R2 , class R3 >
KOKKOS_INLINE_FUNCTION
bool operator != ( const View<LT,L1,L2,L3> & lhs ,
                   const View<RT,R1,R2,R3> & rhs )
{
  return ! ( operator==(lhs,rhs) );
}

} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

template< class OutputView , typename Enable = void >
struct ViewFill {

  using const_value_type = typename OutputView::const_value_type ;

  const OutputView output ;
  const_value_type input ;

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_t i0 ) const
  {
    const size_t n1 = output.dimension_1();
    const size_t n2 = output.dimension_2();
    const size_t n3 = output.dimension_3();
    const size_t n4 = output.dimension_4();
    const size_t n5 = output.dimension_5();
    const size_t n6 = output.dimension_6();
    const size_t n7 = output.dimension_7();

    for ( size_t i1 = 0 ; i1 < n1 ; ++i1 ) {
    for ( size_t i2 = 0 ; i2 < n2 ; ++i2 ) {
    for ( size_t i3 = 0 ; i3 < n3 ; ++i3 ) {
    for ( size_t i4 = 0 ; i4 < n4 ; ++i4 ) {
    for ( size_t i5 = 0 ; i5 < n5 ; ++i5 ) {
    for ( size_t i6 = 0 ; i6 < n6 ; ++i6 ) {
    for ( size_t i7 = 0 ; i7 < n7 ; ++i7 ) {
      output(i0,i1,i2,i3,i4,i5,i6,i7) = input ;
    }}}}}}}
  }

  ViewFill( const OutputView & arg_out , const_value_type & arg_in )
    : output( arg_out ), input( arg_in )
    {
      using execution_space = typename OutputView::execution_space ;
      Kokkos::RangePolicy< execution_space > range( 0 , output.dimension_0() );
      Kokkos::parallel_for( range , *this );
      execution_space::fence();
    }
};

template< class OutputView >
struct ViewFill< OutputView , typename std::enable_if< OutputView::Rank == 0 >::type > {
  ViewFill( const OutputView & dst , const typename OutputView::const_value_type & src )
    {
      Kokkos::Impl::DeepCopy< typename OutputView::memory_space , Kokkos::HostSpace >
        ( dst.data() , & src , sizeof(typename OutputView::const_value_type) );
    }
};

template< class OutputView , class InputView >
struct ViewRemap {

  const OutputView output ;
  const InputView  input ;
  const size_t n0 ;
  const size_t n1 ;
  const size_t n2 ;
  const size_t n3 ;
  const size_t n4 ;
  const size_t n5 ;
  const size_t n6 ;
  const size_t n7 ;

  ViewRemap( const OutputView & arg_out , const InputView & arg_in )
    : output( arg_out ), input( arg_in )
    , n0( std::min( (size_t)arg_out.dimension_0() , (size_t)arg_in.dimension_0() ) )
    , n1( std::min( (size_t)arg_out.dimension_1() , (size_t)arg_in.dimension_1() ) )
    , n2( std::min( (size_t)arg_out.dimension_2() , (size_t)arg_in.dimension_2() ) )
    , n3( std::min( (size_t)arg_out.dimension_3() , (size_t)arg_in.dimension_3() ) )
    , n4( std::min( (size_t)arg_out.dimension_4() , (size_t)arg_in.dimension_4() ) )
    , n5( std::min( (size_t)arg_out.dimension_5() , (size_t)arg_in.dimension_5() ) )
    , n6( std::min( (size_t)arg_out.dimension_6() , (size_t)arg_in.dimension_6() ) )
    , n7( std::min( (size_t)arg_out.dimension_7() , (size_t)arg_in.dimension_7() ) )
    {
      typedef typename OutputView::execution_space execution_space ;
      Kokkos::RangePolicy< execution_space > range( 0 , n0 );
      Kokkos::parallel_for( range , *this );
    }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_t i0 ) const
  {
    for ( size_t i1 = 0 ; i1 < n1 ; ++i1 ) {
    for ( size_t i2 = 0 ; i2 < n2 ; ++i2 ) {
    for ( size_t i3 = 0 ; i3 < n3 ; ++i3 ) {
    for ( size_t i4 = 0 ; i4 < n4 ; ++i4 ) {
    for ( size_t i5 = 0 ; i5 < n5 ; ++i5 ) {
    for ( size_t i6 = 0 ; i6 < n6 ; ++i6 ) {
    for ( size_t i7 = 0 ; i7 < n7 ; ++i7 ) {
      output(i0,i1,i2,i3,i4,i5,i6,i7) = input(i0,i1,i2,i3,i4,i5,i6,i7);
    }}}}}}}
  }
};

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

/** \brief  Deep copy a value from Host memory into a view.  */
template< class DT , class D1 , class D2 , class D3 >
inline
void deep_copy( const View<DT,D1,D2,D3> & dst
              , typename ViewTraits<DT,D1,D2,D3>::const_value_type & value )
{
  static_assert( std::is_same< typename ViewTraits<DT,D1,D2,D3>::non_const_value_type ,
                               typename ViewTraits<DT,D1,D2,D3>::value_type >::value
               , "ERROR: Incompatible deep_copy( View , value )" );

  Kokkos::Experimental::Impl::ViewFill< View<DT,D1,D2,D3> >( dst , value );
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template< class ST , class S1 , class S2 , class S3 >
inline
void deep_copy( ST & dst , const View<ST,S1,S2,S3> & src )
{
  static_assert( ViewTraits<ST,S1,S2,S3>::rank == 0 
               , "ERROR: Non-rank-zero view in deep_copy( value , View )" );

  using src_traits       = ViewTraits<ST,S1,S2,S3> ;
  using src_memory_space = typename src_traits::memory_space ;
  Kokkos::Experimental::Impl::DeepCopy< HostSpace , src_memory_space >( & dst , src.data() , sizeof(ST) );
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type, and rank zero.  */
template< class DT , class D1 , class D2 , class D3
        , class ST , class S1 , class S2 , class S3 >
inline
void deep_copy( const View<DT,D1,D2,D3> & dst ,
                const View<ST,S1,S2,S3> & src ,
                typename std::enable_if<(
                  // Rank zero:
                  ( unsigned(ViewTraits<DT,D1,D2,D3>::rank) == unsigned(0) ) &&
                  ( unsigned(ViewTraits<ST,S1,S2,S3>::rank) == unsigned(0) ) &&
                  // Same type and destination is not constant:
                  std::is_same< typename ViewTraits<DT,D1,D2,D3>::value_type ,
                                typename ViewTraits<ST,S1,S2,S3>::non_const_value_type >::value
                )>::type * = 0 )
{
  using dst_type = View<DT,D1,D2,D3> ;
  using src_type = View<ST,S1,S2,S3> ;

  using value_type       = typename dst_type::value_type ;
  using dst_memory_space = typename dst_type::memory_space ;
  using src_memory_space = typename src_type::memory_space ;

  if ( dst.data() != src.data() ) {
    Kokkos::Impl::DeepCopy< dst_memory_space , src_memory_space >( dst.data() , src.data() , sizeof(value_type) );
  }
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the default specialization, compatible type,
 *          same non-zero rank, same contiguous layout.
 */
template< class DT , class D1 , class D2 , class D3 ,
          class ST , class S1 , class S2 , class S3 >
inline
void deep_copy( const View<DT,D1,D2,D3> & dst ,
                const View<ST,S1,S2,S3> & src ,
                typename std::enable_if<(
                  // Same type and destination is not constant:
                  std::is_same< typename ViewTraits<DT,D1,D2,D3>::value_type ,
                                typename ViewTraits<ST,S1,S2,S3>::non_const_value_type >::value
                  &&
                  // Same rank:
                  ( unsigned(ViewTraits<DT,D1,D2,D3>::rank) ==
                    unsigned(ViewTraits<ST,S1,S2,S3>::rank) )
                  &&
                  // Not specialized, default ViewMapping
                  std::is_same< typename ViewTraits<DT,D1,D2,D3>::specialize , void >::value
                  &&
                  std::is_same< typename ViewTraits<ST,S1,S2,S3>::specialize , void >::value
                )>::type * = 0 )
{
  using dst_type = View<DT,D1,D2,D3> ;
  using src_type = View<ST,S1,S2,S3> ;

  using dst_execution_space = typename dst_type::execution_space ;
  using dst_memory_space    = typename dst_type::memory_space ;
  using src_memory_space    = typename src_type::memory_space ;

  enum { DstExecCanAccessSrc =
   Kokkos::Impl::VerifyExecutionCanAccessMemorySpace< typename dst_execution_space::memory_space , src_memory_space >::value };

  if ( dst.data() != src.data() ) {

    // Concern: If overlapping views then a parallel copy will be erroneous.
    // ...

    // If equal layout, equal dimensions, equal extent, and contiguous memory then can byte-wise copy

    if ( std::is_same< typename ViewTraits<DT,D1,D2,D3>::array_layout ,
                       typename ViewTraits<ST,S1,S2,S3>::array_layout >::value &&
         dst.extent_is_contiguous() &&
         src.extent_is_contiguous() &&
         dst.extent() == src.extent() &&
         dst.dimension_0() == src.dimension_0() &&
         dst.dimension_1() == src.dimension_1() &&
         dst.dimension_2() == src.dimension_2() &&
         dst.dimension_3() == src.dimension_3() &&
         dst.dimension_4() == src.dimension_4() &&
         dst.dimension_5() == src.dimension_5() &&
         dst.dimension_6() == src.dimension_6() &&
         dst.dimension_7() == src.dimension_7() ) {

      const size_t nbytes = sizeof(typename dst_type::value_type) * dst.extent();

      Impl::DeepCopy< dst_memory_space , src_memory_space >( dst.data() , src.data() , nbytes );
    }
    else if ( DstExecCanAccessSrc ) {
      // Copying data between views in accessible memory spaces and either non-contiguous or incompatible shape.
      Impl::ViewRemap< dst_type , src_type >( dst , src );
    }
    else {
      Kokkos::Impl::throw_runtime_exception("deep_copy given views that would require a temporary allocation");
    }
  }
}

} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

template< class T , class A1, class A2, class A3 >
inline
typename Kokkos::Experimental::View<T,A1,A2,A3>::HostMirror
create_mirror( const View<T,A1,A2,A3> & src
             , typename std::enable_if<
                 ! std::is_same< typename ViewTraits<T,A1,A2,A3>::array_layout
                               , Kokkos::LayoutStride >::value
               >::type * = 0
             )
{
  using src_type = View<T,A1,A2,A3> ;
  using dst_type = typename src_type::HostMirror ;

  return dst_type( std::string( src.label() ).append("_mirror")
                 , src.dimension_0()
                 , src.dimension_1()
                 , src.dimension_2()
                 , src.dimension_3()
                 , src.dimension_4()
                 , src.dimension_5()
                 , src.dimension_6()
                 , src.dimension_7() );
}

template< class T , class A1, class A2, class A3 >
inline
typename Kokkos::Experimental::View<T,A1,A2,A3>::HostMirror
create_mirror( const View<T,A1,A2,A3> & src
             , typename std::enable_if<
                 std::is_same< typename ViewTraits<T,A1,A2,A3>::array_layout
                             , Kokkos::LayoutStride >::value
               >::type * = 0
             )
{
  using src_type = View<T,A1,A2,A3> ;
  using dst_type = typename src_type::HostMirror ;

  Kokkos::LayoutStride layout ;

  layout.dimension[0] = src.dimension_0();
  layout.dimension[1] = src.dimension_1();
  layout.dimension[2] = src.dimension_2();
  layout.dimension[3] = src.dimension_3();
  layout.dimension[4] = src.dimension_4();
  layout.dimension[5] = src.dimension_5();
  layout.dimension[6] = src.dimension_6();
  layout.dimension[7] = src.dimension_7();

  layout.stride[0] = src.stride_0();
  layout.stride[1] = src.stride_1();
  layout.stride[2] = src.stride_2();
  layout.stride[3] = src.stride_3();
  layout.stride[4] = src.stride_4();
  layout.stride[5] = src.stride_5();
  layout.stride[6] = src.stride_6();
  layout.stride[7] = src.stride_7();

  return dst_type( std::string( src.label() ).append("_mirror") , layout );
}

template< class T , class A1 , class A2 , class A3 >
inline
typename Kokkos::Experimental::View<T,A1,A2,A3>::HostMirror
create_mirror_view( const View<T,A1,A2,A3> & src
                  , typename std::enable_if<(
                      std::is_same< typename ViewTraits<T,A1,A2,A3>::memory_space
                                  , typename ViewTraits<T,A1,A2,A3>::HostMirror::memory_space
                                  >::value
                    )>::type * = 0 
                  )
{
  return src ;
}

template< class T , class A1 , class A2 , class A3 >
inline
typename Kokkos::Experimental::View<T,A1,A2,A3>::HostMirror
create_mirror_view( const View<T,A1,A2,A3> & src
                  , typename std::enable_if<(
                      ! std::is_same< typename ViewTraits<T,A1,A2,A3>::memory_space
                                    , typename ViewTraits<T,A1,A2,A3>::HostMirror::memory_space
                                    >::value
                    )>::type * = 0 
                  )
{
  return create_mirror( src ); ;
}

} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

/** \brief  Resize a view with copying old data to new data at the corresponding indices. */
template< class T , class A1 , class A2 , class A3 >
inline
void resize( View<T,A1,A2,A3> & v ,
             const size_t n0 = 0 ,
             const size_t n1 = 0 ,
             const size_t n2 = 0 ,
             const size_t n3 = 0 ,
             const size_t n4 = 0 ,
             const size_t n5 = 0 ,
             const size_t n6 = 0 ,
             const size_t n7 = 0 )
{
  using view_type = View<T,A1,A2,A3> ;

  static_assert( ViewTraits<T,A1,A2,A3>::is_managed , "Can only resize managed views" );

  view_type v_resized( v.label(), n0, n1, n2, n3, n4, n5, n6, n7 );

  Kokkos::Experimental::Impl::ViewRemap< view_type , view_type >( v_resized , v );

  v = v_resized ;
}

/** \brief  Resize a view with copying old data to new data at the corresponding indices. */
template< class T , class A1 , class A2 , class A3 >
inline
void realloc( View<T,A1,A2,A3> & v ,
              const size_t n0 = 0 ,
              const size_t n1 = 0 ,
              const size_t n2 = 0 ,
              const size_t n3 = 0 ,
              const size_t n4 = 0 ,
              const size_t n5 = 0 ,
              const size_t n6 = 0 ,
              const size_t n7 = 0 )
{
  using view_type = View<T,A1,A2,A3> ;

  static_assert( ViewTraits<T,A1,A2,A3>::is_managed , "Can only realloc managed views" );

  const std::string label = v.label();

  v = view_type(); // Deallocate first, if the only view to allocation
  v = view_type( label, n0, n1, n2, n3, n4, n5, n6, n7 );
}

} /* namespace Experimental */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#if 0

#include <impl/Kokkos_Atomic_View.hpp>

#endif /* #if 0 */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif

