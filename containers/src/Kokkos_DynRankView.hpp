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

/// \file Kokkos_DynRankView.hpp
/// \brief Declaration and definition of Kokkos::Experimental::DynRankView.
///
/// This header file declares and defines Kokkos::Experimental::DynRankView and its
/// related nonmember functions.

#ifndef KOKKOS_DYNRANKVIEW_HPP
#define KOKKOS_DYNRANKVIEW_HPP

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Error.hpp>
#include <type_traits>

namespace Kokkos {
namespace Experimental {

/* \class DynRankView
 * \brief Container that creates a Kokkos view with runtime rank. 
 *   Essentially this is a rank 8 view with that wraps the access operators
 *   to yield the functionality of a view with rank that varies. 
 */

template< typename DataType , class ... Properties >
class DynRankView : private View< DataType********, Properties... >
{
  static_assert( !std::is_array<DataType>::value && !std::is_pointer<DataType>::value , "Cannot template DynRankView with array or pointer datatype - must be pod" );

public: 
  using view_type = View< DataType******** , Properties...>;
  using reference_type = typename view_type::reference_type; 

private: 
  template < class , class ... > friend class DynRankView ;
  unsigned m_rank;

public:
  KOKKOS_INLINE_FUNCTION
  view_type & DownCast() const { return static_cast< view_type & > (*this); }
  KOKKOS_INLINE_FUNCTION
  const view_type & ConstDownCast() const { return static_cast< const view_type & > (*this); }

  typedef ViewTraits< DataType , Properties ... > traits ;

  typedef typename traits::execution_space execution_space;
  typedef typename traits::memory_space memory_space;

  typedef typename traits::data_type data_type;
  typedef typename traits::const_data_type const_data_type;
  typedef typename traits::non_const_data_type non_const_data_type;

  typedef typename traits::host_mirror_space host_mirror_space;

  /** \brief  Compatible view of array of scalar types */
  typedef DynRankView< typename traits::scalar_array_type ,
                       typename traits::array_layout ,
                typename traits::device_type ,
                typename traits::memory_traits >
    array_type ;

  /** \brief  Compatible view of const data type */
  typedef DynRankView< typename traits::const_data_type ,
                typename traits::array_layout ,
                typename traits::device_type ,
                typename traits::memory_traits >
    const_type ;

  /** \brief  Compatible view of non-const data type */
  typedef DynRankView< typename traits::non_const_data_type ,
                typename traits::array_layout ,
                typename traits::device_type ,
                typename traits::memory_traits >
    non_const_type ;

  /** \brief  Compatible HostMirror view */
  typedef DynRankView< typename traits::non_const_data_type ,
                typename traits::array_layout ,
                typename traits::host_mirror_space >
    HostMirror ;

  //----------------------------------------
  // Domain rank and extents

  KOKKOS_INLINE_FUNCTION
  DynRankView() : view_type() , m_rank(0) {}

  KOKKOS_INLINE_FUNCTION
  constexpr unsigned rank() const { return m_rank; }

  using view_type::extent; 
  using view_type::extent_int; 
  using view_type::layout;
  using view_type::dimension;
  using view_type::size;
  using view_type::stride;

  using pointer_type = typename view_type::pointer_type;
  using view_type::reference_type_is_lvalue_reference;
  using view_type::span;
  using view_type::capacity;
  using view_type::span_is_contiguous;
  using view_type::data;
  using view_type::implementation_map;

  using view_type::is_contiguous;
  using view_type::ptr_on_device;

  //Deprecated, remove soon (add for test)
  using view_type::dimension_0;
  using view_type::dimension_1;
  using view_type::dimension_2;
  using view_type::dimension_3;
  using view_type::dimension_4;
  using view_type::dimension_5;
  using view_type::dimension_6;
  using view_type::dimension_7;
  using view_type::stride_0;
  using view_type::stride_1;
  using view_type::stride_2;
  using view_type::stride_3;
  using view_type::stride_4;
  using view_type::stride_5;
  using view_type::stride_6;
  using view_type::stride_7;

  //operators ()
  // Rank 0
  KOKKOS_INLINE_FUNCTION
  reference_type operator()() const
    { return view_type::operator()(0,0,0,0,0,0,0,0); }
  
  // Rank 1
  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  reference_type operator[](const iType i0) const
    { return view_type::operator[](i0,0,0,0,0,0,0,0); }

  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType i0 ) const 
    { return view_type::operator()(i0,0,0,0,0,0,0,0); }

  // Rank 2
  template< typename iType0 , typename iType1 >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType0 i0 , const iType1 i1 ) const 
    { return view_type::operator()(i0,i1,0,0,0,0,0,0); }

  // Rank 3
  template< typename iType0 , typename iType1 , typename iType2 >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType0 i0 , const iType1 i1 , const iType2 i2 ) const 
    { return view_type::operator()(i0,i1,i2,0,0,0,0,0); }

  // Rank 4
  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType0 i0 , const iType1 i1 , const iType2 i2 , const iType3 i3 ) const 
    { return view_type::operator()(i0,i1,i2,i3,0,0,0,0); }

  // Rank 5
  template< typename iType0 , typename iType1 , typename iType2 , typename iType3, typename iType4 >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType0 i0 , const iType1 i1 , const iType2 i2 , const iType3 i3 , const iType4 i4 ) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,0,0,0); }

  // Rank 6
  template< typename iType0 , typename iType1 , typename iType2 , typename iType3, typename iType4 , typename iType5 >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType0 i0 , const iType1 i1 , const iType2 i2 , const iType3 i3 , const iType4 i4 , const iType5 i5 ) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,i5,0,0); }

  // Rank 7
  template< typename iType0 , typename iType1 , typename iType2 , typename iType3, typename iType4 , typename iType5 , typename iType6 >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType0 i0 , const iType1 i1 , const iType2 i2 , const iType3 i3 , const iType4 i4 , const iType5 i5 , const iType6 i6 ) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,i5,i6,0); }

  // Rank 8
  template< typename iType0 , typename iType1 , typename iType2 , typename iType3, typename iType4 , typename iType5 , typename iType6 , typename iType7 >
  KOKKOS_INLINE_FUNCTION
  reference_type operator()(const iType0 i0 , const iType1 i1 , const iType2 i2 , const iType3 i3 , const iType4 i4 , const iType5 i5 , const iType6 i6 , const iType7 i7 ) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,i5,i6,i7); }


  //----------------------------------------
  // Standard constructor, destructor, and assignment operators... 

  KOKKOS_INLINE_FUNCTION
  ~DynRankView() {}

  KOKKOS_INLINE_FUNCTION
  DynRankView( const DynRankView & ) = default ;

  KOKKOS_INLINE_FUNCTION
  DynRankView( DynRankView && ) = default ;

  KOKKOS_INLINE_FUNCTION
  DynRankView & operator = ( const DynRankView & ) = default ;

  KOKKOS_INLINE_FUNCTION
  DynRankView & operator = ( DynRankView && ) = default ;

  //----------------------------------------
  // Compatible view copy constructor and assignment
  // may assign unmanaged from managed.

  template< class RT , class ... RP >
  KOKKOS_INLINE_FUNCTION
  DynRankView( const DynRankView<RT,RP...> & rhs )
    : view_type( rhs.ConstDownCast() )  
    , m_rank(rhs.m_rank)
    {}

  template< class RT , class ... RP >
  KOKKOS_INLINE_FUNCTION
  DynRankView & operator = (const DynRankView<RT,RP...> & rhs )
  {
    view_type::operator = ( rhs.ConstDownCast() );
    m_rank = rhs.rank();
    return *this;
  }

  //----------------------------------------
  // Compatible subview constructor
  // may assign unmanaged from managed.
  // ...

  //----------------------------------------
  // Allocation tracking properties

  using view_type::use_count;
  using view_type::label;

  //----------------------------------------
  // Allocation according to allocation properties and array layout

  template< class ... P >
  explicit inline
  DynRankView( const Impl::ViewCtorProp< P ... > & arg_prop
      , typename std::enable_if< ! Impl::ViewCtorProp< P... >::has_pointer
                               , typename traits::array_layout
                               >::type const & arg_layout
      )
      : view_type( arg_prop 
                 , typename traits::array_layout
                   ( arg_layout.dimension[0] != 0 ? arg_layout.dimension[0] : 1 
                   , arg_layout.dimension[1] != 0 ? arg_layout.dimension[1] : 1 
                   , arg_layout.dimension[2] != 0 ? arg_layout.dimension[2] : 1 
                   , arg_layout.dimension[3] != 0 ? arg_layout.dimension[3] : 1 
                   , arg_layout.dimension[4] != 0 ? arg_layout.dimension[4] : 1 
                   , arg_layout.dimension[5] != 0 ? arg_layout.dimension[5] : 1 
                   , arg_layout.dimension[6] != 0 ? arg_layout.dimension[6] : 1 
                   , arg_layout.dimension[7] != 0 ? arg_layout.dimension[7] : 1 
                   ) 
                 )
      , m_rank( ( arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0 && arg_layout.dimension[2] == 0 && arg_layout.dimension[1] == 0 && arg_layout.dimension[0] == 0) ? 0 
            : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0 && arg_layout.dimension[2] == 0 && arg_layout.dimension[1] == 0) ? 1 
            : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0 && arg_layout.dimension[2] == 0) ? 2 
            : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0) ? 3 
            : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0) ? 4 
            : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0) ? 5 
            : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0) ? 6 
            : ( arg_layout.dimension[7] == 0 ? 7 
            : 8 ) ) ) ) ) ) )
            )  
    {}

//Wrappers
  template< class ... P >
  explicit KOKKOS_INLINE_FUNCTION
  DynRankView( const Impl::ViewCtorProp< P ... > & arg_prop
      , typename std::enable_if< Impl::ViewCtorProp< P... >::has_pointer
                               , typename traits::array_layout
                               >::type const & arg_layout
      )
      : view_type( arg_prop 
                , typename traits::array_layout
                   ( arg_layout.dimension[0] != 0 ? arg_layout.dimension[0] : 1 
                   , arg_layout.dimension[1] != 0 ? arg_layout.dimension[1] : 1 
                   , arg_layout.dimension[2] != 0 ? arg_layout.dimension[2] : 1 
                   , arg_layout.dimension[3] != 0 ? arg_layout.dimension[3] : 1 
                   , arg_layout.dimension[4] != 0 ? arg_layout.dimension[4] : 1 
                   , arg_layout.dimension[5] != 0 ? arg_layout.dimension[5] : 1 
                   , arg_layout.dimension[6] != 0 ? arg_layout.dimension[6] : 1 
                   , arg_layout.dimension[7] != 0 ? arg_layout.dimension[7] : 1 
                   ) 
               )
      , m_rank( 
               ( arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0 && arg_layout.dimension[2] == 0 && arg_layout.dimension[1] == 0 && arg_layout.dimension[0] == 0) ? 0 
              : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0 && arg_layout.dimension[2] == 0 && arg_layout.dimension[1] == 0) ? 1 
              : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0 && arg_layout.dimension[2] == 0) ? 2 
              : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0 && arg_layout.dimension[3] == 0) ? 3 
              : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0 && arg_layout.dimension[4] == 0) ? 4 
              : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0 && arg_layout.dimension[5] == 0) ? 5 
              : ( (arg_layout.dimension[7] == 0 && arg_layout.dimension[6] == 0) ? 6 
              : ( arg_layout.dimension[7] == 0 ? 7 
              : 8 ) ) ) ) ) ) )
            )  
    {}

  //----------------------------------------
  //Constructor(s)

  // Simple dimension-only layout
  template< class ... P >
  explicit inline
  DynRankView( const Impl::ViewCtorProp< P ... > & arg_prop
      , typename std::enable_if< ! Impl::ViewCtorProp< P... >::has_pointer
                               , size_t
                               >::type const arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0
      )
    : DynRankView( arg_prop 
    , typename traits::array_layout
          ( arg_N0 , arg_N1 , arg_N2 , arg_N3 , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
      )
    {}

  template< class ... P >
  explicit KOKKOS_INLINE_FUNCTION
  DynRankView( const Impl::ViewCtorProp< P ... > & arg_prop
      , typename std::enable_if< Impl::ViewCtorProp< P... >::has_pointer
                               , size_t
                               >::type const arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0
      )
    : DynRankView( arg_prop 
    , typename traits::array_layout
          ( arg_N0 , arg_N1 , arg_N2 , arg_N3 , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
      )
    {}

  // Allocate with label and layout
  template< typename Label >
  explicit inline
  DynRankView( const Label & arg_label
      , typename std::enable_if<
          Kokkos::Experimental::Impl::is_view_label<Label>::value ,
          typename traits::array_layout >::type const & arg_layout
      )
    : DynRankView( Impl::ViewCtorProp< std::string >( arg_label ) , arg_layout )
    {}

  // Allocate label and layout, must disambiguate from subview constructor.
  template< typename Label >
  explicit inline
  DynRankView( const Label & arg_label
      , typename std::enable_if<
          Kokkos::Experimental::Impl::is_view_label<Label>::value ,
        const size_t >::type arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0
      )
    : DynRankView( Impl::ViewCtorProp< std::string >( arg_label )
    , typename traits::array_layout
          ( arg_N0 , arg_N1 , arg_N2 , arg_N3 , arg_N4 , arg_N5 , arg_N6 , arg_N7 )
          )
    {}

  // For backward compatibility
/*
  explicit inline
  DynRankView( const ViewAllocateWithoutInitializing & arg_prop
      , const typename traits::array_layout & arg_layout
      )
    : view_type( Impl::ViewCtorProp< std::string , Kokkos::Experimental::Impl::WithoutInitializing_t >( arg_prop.label , Kokkos::Experimental::WithoutInitializing )
          , arg_layout
          )
    //, m_rank(arg_N0 == 0 ? 0 : ( arg_N1 == 0 ? 1 : ( arg_N2 == 0 ? 2 : ( arg_N3 == 0 ? 3 : ( arg_N4 == 0 ? 4 : ( arg_N5 == 0 ? 5 : ( arg_N6 == 0 ? 6 : ( arg_N7 == 0 ? 7 : 8 ) ) ) ) ) ) ) ) //how to extract rank?
    {}
*/

  explicit inline
  DynRankView( const ViewAllocateWithoutInitializing & arg_prop
      , const size_t arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0
      )
    : DynRankView(Impl::ViewCtorProp< std::string , Kokkos::Experimental::Impl::WithoutInitializing_t >( arg_prop.label , Kokkos::Experimental::WithoutInitializing ), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7 ) 
    {}

  using view_type::memory_span;

  explicit KOKKOS_INLINE_FUNCTION
  DynRankView( pointer_type arg_ptr
      , const size_t arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0
      )
    : DynRankView( Impl::ViewCtorProp<pointer_type>(arg_ptr) , arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7 )
    {}

  explicit KOKKOS_INLINE_FUNCTION
  DynRankView( pointer_type arg_ptr
      , typename traits::array_layout & arg_layout
      )
    : DynRankView( Impl::ViewCtorProp<pointer_type>(arg_ptr) , arg_layout )
    {}


  //----------------------------------------
  // Shared scratch memory constructor

  using view_type::shmem_size; 

  explicit KOKKOS_INLINE_FUNCTION
  DynRankView( const typename traits::execution_space::scratch_memory_space & arg_space
      , const size_t arg_N0 = 0
      , const size_t arg_N1 = 0
      , const size_t arg_N2 = 0
      , const size_t arg_N3 = 0
      , const size_t arg_N4 = 0
      , const size_t arg_N5 = 0
      , const size_t arg_N6 = 0
      , const size_t arg_N7 = 0 )
    : view_type( arg_space
                 , arg_N0 != 0 ? arg_N0 : 1 
                 , arg_N1 != 0 ? arg_N1 : 1 
                 , arg_N2 != 0 ? arg_N2 : 1 
                 , arg_N3 != 0 ? arg_N3 : 1 
                 , arg_N4 != 0 ? arg_N4 : 1 
                 , arg_N5 != 0 ? arg_N5 : 1 
                 , arg_N6 != 0 ? arg_N6 : 1 
                 , arg_N7 != 0 ? arg_N7 : 1 
               )
    , m_rank( 
             ( arg_N7 == 0 && arg_N6 == 0 && arg_N5 == 0 && arg_N4 == 0 && arg_N3 == 0 && arg_N2 == 0 && arg_N1 == 0 && arg_N0 == 0) ? 0 
             : ( (arg_N7 == 0 && arg_N6 == 0 && arg_N5 == 0 && arg_N4 == 0 && arg_N3 == 0 && arg_N2 == 0 && arg_N1 == 0) ? 1 
             : ( (arg_N7 == 0 && arg_N6 == 0 && arg_N5 == 0 && arg_N4 == 0 && arg_N3 == 0 && arg_N2 == 0) ? 2 
             : ( (arg_N7 == 0 && arg_N6 == 0 && arg_N5 == 0 && arg_N4 == 0 && arg_N3 == 0) ? 3 
             : ( (arg_N7 == 0 && arg_N6 == 0 && arg_N5 == 0 && arg_N4 == 0) ? 4 
             : ( (arg_N7 == 0 && arg_N6 == 0 && arg_N5 == 0) ? 5 
             : ( (arg_N7 == 0 && arg_N6 == 0) ? 6 
             : ( arg_N7 == 0 ? 7 
             : 8 ) ) ) ) ) ) ) 
            ) 
    {}

  //----------------------------------------
  //using Subview...

};

} // namespace Experimental
} // namespace Kokkos


namespace Kokkos {
namespace Experimental {

// overload == and !=
template< class LT , class ... LP , class RT , class ... RP >
KOKKOS_INLINE_FUNCTION
bool operator == ( const DynRankView<LT,LP...> & lhs ,
                   const DynRankView<RT,RP...> & rhs )
{
  // Same data, layout, dimensions
  typedef ViewTraits<LT,LP...>  lhs_traits ;
  typedef ViewTraits<RT,RP...>  rhs_traits ;

  return
    std::is_same< typename lhs_traits::const_value_type ,
                  typename rhs_traits::const_value_type >::value &&
    std::is_same< typename lhs_traits::array_layout ,
                  typename rhs_traits::array_layout >::value &&
    std::is_same< typename lhs_traits::memory_space ,
                  typename rhs_traits::memory_space >::value &&
    lhs.rank()       ==  rhs.rank() &&
    lhs.data()       == rhs.data() &&
    lhs.span()       == rhs.span() &&
    lhs.dimension(0) == rhs.dimension(0) &&
    lhs.dimension(1) == rhs.dimension(1) &&
    lhs.dimension(2) == rhs.dimension(2) &&
    lhs.dimension(3) == rhs.dimension(3) &&
    lhs.dimension(4) == rhs.dimension(4) &&
    lhs.dimension(5) == rhs.dimension(5) &&
    lhs.dimension(6) == rhs.dimension(6) &&
    lhs.dimension(7) == rhs.dimension(7);
}

template< class LT , class ... LP , class RT , class ... RP >
KOKKOS_INLINE_FUNCTION
bool operator != ( const DynRankView<LT,LP...> & lhs ,
                   const DynRankView<RT,RP...> & rhs )
{
  return ! ( operator==(lhs,rhs) );
}

} //end Experimental
} //end Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

/** \brief  Deep copy a value from Host memory into a view.  */
template< class DT , class ... DP >
inline
void deep_copy
  ( const DynRankView<DT,DP...> & dst
  , typename ViewTraits<DT,DP...>::const_value_type & value
  , typename std::enable_if<
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value
    >::type * = 0 )
{
  deep_copy( dst.ConstDownCast() , value );
}

/** \brief  Deep copy into a value in Host memory from a view.  */
template< class ST , class ... SP >
inline
void deep_copy
  ( typename ViewTraits<ST,SP...>::non_const_value_type & dst
  , const DynRankView<ST,SP...> & src
  , typename std::enable_if<
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value
    >::type * = 0 )
{
  deep_copy( dst , src.ConstDownCast() );
}


//----------------------------------------------------------------------------
/** \brief  A deep copy between views of compatible type */
template< class DT , class ... DP , class ST , class ... SP >
inline
void deep_copy
  ( const DynRankView<DT,DP...> & dst
  , const DynRankView<ST,SP...> & src
  , typename std::enable_if<(
    std::is_same< typename ViewTraits<DT,DP...>::specialize , void >::value &&
    std::is_same< typename ViewTraits<ST,SP...>::specialize , void >::value // &&
  )>::type * = 0 )
{
  deep_copy( dst.ConstDownCast() , src.ConstDownCast() );
}

} //end Experimental
} //end Kokkos


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

template< class T , class ... P >
inline
typename DynRankView<T,P...>::HostMirror
create_mirror( const DynRankView<T,P...> & src
             , typename std::enable_if<
                 ! std::is_same< typename Kokkos::Experimental::ViewTraits<T,P...>::array_layout
                               , Kokkos::LayoutStride >::value
               >::type * = 0
             )
{
  typedef DynRankView<T,P...>                   src_type ;
  typedef typename src_type::HostMirror  dst_type ;

  return dst_type( std::string( src.label() ).append("_mirror")
                 , src.dimension(0)
                 , src.dimension(1)
                 , src.dimension(2)
                 , src.dimension(3)
                 , src.dimension(4)
                 , src.dimension(5)
                 , src.dimension(6)
                 , src.dimension(7) );
}


template< class T , class ... P >
inline
typename DynRankView<T,P...>::HostMirror
create_mirror( const DynRankView<T,P...> & src
             , typename std::enable_if<
                 std::is_same< typename Kokkos::Experimental::ViewTraits<T,P...>::array_layout
                             , Kokkos::LayoutStride >::value
               >::type * = 0
             )
{
  typedef DynRankView<T,P...>                   src_type ;
  typedef typename src_type::HostMirror  dst_type ;

  Kokkos::LayoutStride layout ;

  layout.dimension[0] = src.dimension(0);
  layout.dimension[1] = src.dimension(1);
  layout.dimension[2] = src.dimension(2);
  layout.dimension[3] = src.dimension(3);
  layout.dimension[4] = src.dimension(4);
  layout.dimension[5] = src.dimension(5);
  layout.dimension[6] = src.dimension(6);
  layout.dimension[7] = src.dimension(7);

  layout.stride[0] = src.stride(0);
  layout.stride[1] = src.stride(1);
  layout.stride[2] = src.stride(2);
  layout.stride[3] = src.stride(3);
  layout.stride[4] = src.stride(4);
  layout.stride[5] = src.stride(5);
  layout.stride[6] = src.stride(6);
  layout.stride[7] = src.stride(7);

  return dst_type( std::string( src.label() ).append("_mirror") , layout );
}

template< class T , class ... P >
inline
typename DynRankView<T,P...>::HostMirror
create_mirror_view( const DynRankView<T,P...> & src
                  , typename std::enable_if<(
                      std::is_same< typename DynRankView<T,P...>::memory_space
                                  , typename DynRankView<T,P...>::HostMirror::memory_space
                                  >::value
                      &&
                      std::is_same< typename DynRankView<T,P...>::data_type
                                  , typename DynRankView<T,P...>::HostMirror::data_type
                                  >::value
                    )>::type * = 0
                  )
{
  return src ;
}

template< class T , class ... P >
inline
typename DynRankView<T,P...>::HostMirror
create_mirror_view( const DynRankView<T,P...> & src
                  , typename std::enable_if< ! (
                      std::is_same< typename DynRankView<T,P...>::memory_space
                                  , typename DynRankView<T,P...>::HostMirror::memory_space
                                  >::value
                      &&
                      std::is_same< typename DynRankView<T,P...>::data_type
                                  , typename DynRankView<T,P...>::HostMirror::data_type
                                  >::value
                    )>::type * = 0
                  )
{
  return Kokkos::Experimental::create_mirror( src ); 
}

} //end Experimental
} //end Kokkos


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {

/** \brief  Resize a view with copying old data to new data at the corresponding indices. */
template< class T , class ... P >
inline
void resize( DynRankView<T,P...> & v ,
             const size_t n0 = 0 ,
             const size_t n1 = 0 ,
             const size_t n2 = 0 ,
             const size_t n3 = 0 ,
             const size_t n4 = 0 ,
             const size_t n5 = 0 ,
             const size_t n6 = 0 ,
             const size_t n7 = 0 )
{
  typedef DynRankView<T,P...>  drview_type ;

  static_assert( Kokkos::Experimental::ViewTraits<T,P...>::is_managed , "Can only resize managed views" );

  drview_type v_resized( v.label(), n0, n1, n2, n3, n4, n5, n6, n7 );

  Kokkos::Experimental::Impl::ViewRemap< drview_type , drview_type >( v_resized , v );

  v = v_resized ;
}

/** \brief  Resize a view with copying old data to new data at the corresponding indices. */
template< class T , class ... P >
inline
void realloc( DynRankView<T,P...> & v ,
              const size_t n0 = 0 ,
              const size_t n1 = 0 ,
              const size_t n2 = 0 ,
              const size_t n3 = 0 ,
              const size_t n4 = 0 ,
              const size_t n5 = 0 ,
              const size_t n6 = 0 ,
              const size_t n7 = 0 )
{
  typedef DynRankView<T,P...>  view_type ;

  static_assert( Kokkos::Experimental::ViewTraits<T,P...>::is_managed , "Can only realloc managed views" );

  const std::string label = v.label();

  v = view_type(); // Deallocate first, if the only view to allocation
  v = view_type( label, n0, n1, n2, n3, n4, n5, n6, n7 );
}

} //end Experimental
} //end Kokkos

#endif
