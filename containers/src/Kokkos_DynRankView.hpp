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
/// \brief Declaration and definition of Kokkos::DynRankView.
///
/// This header file declares and defines Kokkos::DynRankView and its
/// related nonmember functions.

#ifndef KOKKOS_DYNRANKVIEW_HPP
#define KOKKOS_DYNRANKVIEW_HPP

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Error.hpp>
#include <type_traits>

namespace Kokkos {

/* \class DynRankView
 * \brief Container to manage mirroring a Kokkos::View that lives
 *   in device memory with a Kokkos::View that lives in host memory.
 *
 */

namespace impl {
template <typename T>
struct ptr_type {
  typedef T type;
};

template <typename T>
struct ptr_type <T*> {
  typedef T type;
};

} //end impl


template< typename DataType , class ... P >
class DynRankView : private View< DataType********, P... >
{
  static_assert( !std::is_array<DataType>::value && !std::is_pointer<DataType>::value , "Cannot template DynRankView with array or pointer datatype - must be pod" );

public: 
  using view_type = View< DataType******** , P...>;
//  typedef typename view_type::reference_type return_type;
//  typedef typename view_type::reference_type reference_type;
  //using reference_type = typename view_type::reference_type; 
  using typename view_type::reference_type; 
private: 

public:
  //Constructor(s)
  template< typename iType >
  DynRankView(iType dim0 = 0, iType dim1 = 0, iType dim2 = 0, iType dim3 = 0, iType dim4 = 0, iType dim5 = 0, iType dim6 = 0, iType dim7 = 0) : view_type("",dim0,dim1,dim2,dim3,dim4,dim5,dim6,dim7) {}

  //operator ()
  template< typename iType >
  reference_type operator()(const iType i0 ) const 
    { return view_type::operator()(i0,0,0,0,0,0,0,0); }

  template< typename iType >
  reference_type operator()(const iType i0 , const iType i1 ) const 
    { return view_type::operator()(i0,i1,0,0,0,0,0,0); }

  template< typename iType >
  reference_type operator()(const iType i0 , const iType i1 , const iType i2 ) const 
    { return view_type::operator()(i0,i1,i2,0,0,0,0,0); }

  template< typename iType >
  reference_type operator()(const iType i0 , const iType i1 , const iType i2 , const iType i3 ) const 
    { return view_type::operator()(i0,i1,i2,i3,0,0,0,0); }

  template< typename iType >
  reference_type operator()(const iType i0 , const iType i1 , const iType i2 , const iType i3 , const iType i4 ) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,0,0,0); }

  template< typename iType >
  reference_type operator()(const iType i0 , const iType i1 , const iType i2 , const iType i3 , const iType i4 , const iType i5 ) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,i5,0,0); }

  template< typename iType >
  reference_type operator()(const iType i0 , const iType i1 , const iType i2 , const iType i3 , const iType i4 , const iType i5 , const iType i6 ) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,i5,i6,0); }

  template< typename iType >
  reference_type operator()(const iType i0 = 0, const iType i1 = 0, const iType i2 = 0, const iType i3 = 0, const iType i4 = 0, const iType i5 = 0, const iType i6 = 0, const iType i7 = 0) const 
    { return view_type::operator()(i0,i1,i2,i3,i4,i5,i6,i7); }


  //rank
//  enum { Rank = view_type::Rank };
  using view_type::Rank ;

  template< typename iType >
  KOKKOS_INLINE_FUNCTION constexpr
  typename std::enable_if< std::is_integral<iType>::value, size_t >::type
  extent( const iType &r ) const
    { return view_type::extent( r ); }

  template< typename iType >
  KOKKOS_INLINE_FUNCTION constexpr
  typename std::enable_if< std::is_integral<iType>::value, int >::type
  extent_int( const iType &r ) const
    { return view_type::extent_int( r ); }

//  using view_type::extent_int( r ) ; 

  KOKKOS_INLINE_FUNCTION constexpr
  typename view_type::traits::array_layout layout() const
    { return view_type::layout(); }

  template< typename iType >
  KOKKOS_INLINE_FUNCTION constexpr
  typename std::enable_if< std::is_integral<iType>::value, size_t >::type
  dimension( const iType & r ) const { return extent( r ); }

  
  KOKKOS_INLINE_FUNCTION constexpr size_t size() const { return view_type::size(); }

  template< typename iType >
  KOKKOS_INLINE_FUNCTION constexpr void stride( iType * const s ) const { view_type::stride( s ); }

  //Range span ...
//  typedef typename view_type::reference_type reference_type;
  typedef typename view_type::pointer_type   pointer_type;

  enum { reference_type_is_lvalue_reference = std::is_lvalue_reference< reference_type >::value };

  KOKKOS_INLINE_FUNCTION constexpr size_t span() const { return view_type::span(); }

  KOKKOS_INLINE_FUNCTION constexpr size_t capacity() const { return view_type::capacity(); }
  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const { return view_type::span_is_contiguous(); }

  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const { return view_type::data(); }

  KOKKOS_INLINE_FUNCTION
  const Kokkos::Experimental::Impl::ViewMapping< typename view_type::traits , void > &
  implementation_map() const { return view_type::implementation_map(); }

  // Standard constructor, destructor, and assignment operators... 



};

} // namespace Kokkos

#endif
