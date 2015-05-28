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

#include <gtest/gtest.h>

#include <stdexcept>
#include <sstream>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <KokkosExp_View.hpp>

/*--------------------------------------------------------------------------*/

namespace Test {

template< class RangeType >
void test_view_range( const size_t N , const RangeType & range , const size_t begin , const size_t dim )
{
  using query = Kokkos::Experimental::Impl::ViewOffsetRange< RangeType > ;

  ASSERT_EQ( query::begin( range ) , begin );
  ASSERT_EQ( query::dimension( N , range ) , dim );
  ASSERT_EQ( query::is_range , dim != 0 );
}


template< class HostExecSpace >
void test_view_mapping()
{
  typedef Kokkos::Experimental::Impl::ViewDimension<>  dim_0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<2> dim_s2 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<2,3> dim_s2_s3 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<2,3,4> dim_s2_s3_s4 ;

  typedef Kokkos::Experimental::Impl::ViewDimension<0> dim_s0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,3> dim_s0_s3 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,3,4> dim_s0_s3_s4 ;

  typedef Kokkos::Experimental::Impl::ViewDimension<0,0> dim_s0_s0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,0,4> dim_s0_s0_s4 ;

  typedef Kokkos::Experimental::Impl::ViewDimension<0,0,0> dim_s0_s0_s0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,0,0,0> dim_s0_s0_s0_s0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,0,0,0,0> dim_s0_s0_s0_s0_s0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,0,0,0,0,0> dim_s0_s0_s0_s0_s0_s0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,0,0,0,0,0,0> dim_s0_s0_s0_s0_s0_s0_s0 ;
  typedef Kokkos::Experimental::Impl::ViewDimension<0,0,0,0,0,0,0,0> dim_s0_s0_s0_s0_s0_s0_s0_s0 ;

  // Fully static dimensions should not be larger than an int
  ASSERT_LE( sizeof(dim_0) , sizeof(int) );
  ASSERT_LE( sizeof(dim_s2) , sizeof(int) );
  ASSERT_LE( sizeof(dim_s2_s3) , sizeof(int) );
  ASSERT_LE( sizeof(dim_s2_s3_s4) , sizeof(int) );

  // Rank 1 is size_t
  ASSERT_EQ( sizeof(dim_s0) , sizeof(size_t) );
  ASSERT_EQ( sizeof(dim_s0_s3) , sizeof(size_t) );
  ASSERT_EQ( sizeof(dim_s0_s3_s4) , sizeof(size_t) );

  // Allow for padding
  ASSERT_LE( sizeof(dim_s0_s0) , 2 * sizeof(size_t) );
  ASSERT_LE( sizeof(dim_s0_s0_s4) , 2 * sizeof(size_t) );

  ASSERT_LE( sizeof(dim_s0_s0_s0) , 4 * sizeof(size_t) );
  ASSERT_EQ( sizeof(dim_s0_s0_s0_s0) , 4 * sizeof(unsigned) );
  ASSERT_LE( sizeof(dim_s0_s0_s0_s0_s0) , 6 * sizeof(unsigned) );
  ASSERT_EQ( sizeof(dim_s0_s0_s0_s0_s0_s0) , 6 * sizeof(unsigned) );
  ASSERT_LE( sizeof(dim_s0_s0_s0_s0_s0_s0_s0) , 8 * sizeof(unsigned) );
  ASSERT_EQ( sizeof(dim_s0_s0_s0_s0_s0_s0_s0_s0) , 8 * sizeof(unsigned) );

  ASSERT_EQ( int(dim_0::rank) , int(0) );
  ASSERT_EQ( int(dim_0::rank_dynamic) , int(0) );

  ASSERT_EQ( int(dim_s2::rank) , int(1) );
  ASSERT_EQ( int(dim_s2::rank_dynamic) , int(0) );

  ASSERT_EQ( int(dim_s2_s3::rank) , int(2) );
  ASSERT_EQ( int(dim_s2_s3::rank_dynamic) , int(0) );

  ASSERT_EQ( int(dim_s2_s3_s4::rank) , int(3) );
  ASSERT_EQ( int(dim_s2_s3_s4::rank_dynamic) , int(0) );

  ASSERT_EQ( int(dim_s0::rank) , int(1) );
  ASSERT_EQ( int(dim_s0::rank_dynamic) , int(1) );

  ASSERT_EQ( int(dim_s0_s3::rank) , int(2) );
  ASSERT_EQ( int(dim_s0_s3::rank_dynamic) , int(1) );

  ASSERT_EQ( int(dim_s0_s3_s4::rank) , int(3) );
  ASSERT_EQ( int(dim_s0_s3_s4::rank_dynamic) , int(1) );

  ASSERT_EQ( int(dim_s0_s0_s4::rank) , int(3) );
  ASSERT_EQ( int(dim_s0_s0_s4::rank_dynamic) , int(2) );

  ASSERT_EQ( int(dim_s0_s0_s0::rank) , int(3) );
  ASSERT_EQ( int(dim_s0_s0_s0::rank_dynamic) , int(3) );

  ASSERT_EQ( int(dim_s0_s0_s0_s0::rank) , int(4) );
  ASSERT_EQ( int(dim_s0_s0_s0_s0::rank_dynamic) , int(4) );

  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0::rank) , int(5) );
  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0::rank_dynamic) , int(5) );

  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0_s0::rank) , int(6) );
  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0_s0::rank_dynamic) , int(6) );

  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0_s0_s0::rank) , int(7) );
  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0_s0_s0::rank_dynamic) , int(7) );

  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0_s0_s0_s0::rank) , int(8) );
  ASSERT_EQ( int(dim_s0_s0_s0_s0_s0_s0_s0_s0::rank_dynamic) , int(8) );

  dim_s0          d1( 2, 3, 4, 5, 6, 7, 8, 9 ); 
  dim_s0_s0       d2( 2, 3, 4, 5, 6, 7, 8, 9 );
  dim_s0_s0_s0    d3( 2, 3, 4, 5, 6, 7, 8, 9 );
  dim_s0_s0_s0_s0 d4( 2, 3, 4, 5, 6, 7, 8, 9 );

  ASSERT_EQ( d1.N0 , 2 );
  ASSERT_EQ( d2.N0 , 2 );
  ASSERT_EQ( d3.N0 , 2 );
  ASSERT_EQ( d4.N0 , 2 );

  ASSERT_EQ( d1.N1 , 1 );
  ASSERT_EQ( d2.N1 , 3 );
  ASSERT_EQ( d3.N1 , 3 );
  ASSERT_EQ( d4.N1 , 3 );

  ASSERT_EQ( d1.N2 , 1 );
  ASSERT_EQ( d2.N2 , 1 );
  ASSERT_EQ( d3.N2 , 4 );
  ASSERT_EQ( d4.N2 , 4 );

  ASSERT_EQ( d1.N3 , 1 );
  ASSERT_EQ( d2.N3 , 1 );
  ASSERT_EQ( d3.N3 , 1 );
  ASSERT_EQ( d4.N3 , 5 );

  //----------------------------------------

  using stride_s0_s0_s0 = Kokkos::Experimental::Impl::ViewOffset< dim_s0_s0_s0 , Kokkos::LayoutStride > ;

  //----------------------------------------
  // Static dimension
  {
    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s2_s3_s4 , Kokkos::LayoutLeft > left_s2_s3_s4 ;

    ASSERT_EQ( sizeof(left_s2_s3_s4) , sizeof(dim_s2_s3_s4) );

    left_s2_s3_s4 off3 ;

    stride_s0_s0_s0  stride3( off3 );

    ASSERT_EQ( off3.stride_0() , 1 );
    ASSERT_EQ( off3.stride_1() , 2 );
    ASSERT_EQ( off3.stride_2() , 6 );
    ASSERT_EQ( off3.extent() , 24 );

    ASSERT_EQ( off3.stride_0() , stride3.stride_0() );
    ASSERT_EQ( off3.stride_1() , stride3.stride_1() );
    ASSERT_EQ( off3.stride_2() , stride3.stride_2() );
    ASSERT_EQ( off3.extent() , stride3.extent() );

    int offset = 0 ;

    for ( int k = 0 ; k < 4 ; ++k ){
    for ( int j = 0 ; j < 3 ; ++j ){
    for ( int i = 0 ; i < 2 ; ++i , ++offset ){
      ASSERT_EQ( off3(i,j,k) , offset );
      ASSERT_EQ( stride3(i,j,k) , off3(i,j,k) );
    }}}
  }

  //----------------------------------------
  // Small dimension is unpadded
  {
    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s0_s0_s4 , Kokkos::LayoutLeft > left_s0_s0_s4 ;

    left_s0_s0_s4 dyn_off3( std::integral_constant<unsigned,sizeof(int)>(), 2, 3, 0, 0, 0, 0, 0, 0 );

    stride_s0_s0_s0  stride3( dyn_off3 );

    ASSERT_EQ( dyn_off3.m_dim.rank , 3 );
    ASSERT_EQ( dyn_off3.m_dim.N0 , 2 );
    ASSERT_EQ( dyn_off3.m_dim.N1 , 3 );
    ASSERT_EQ( dyn_off3.m_dim.N2 , 4 );
    ASSERT_EQ( dyn_off3.m_dim.N3 , 1 );
    ASSERT_EQ( dyn_off3.size() , 2 * 3 * 4 );

    ASSERT_EQ( stride3.m_dim.rank , 3 );
    ASSERT_EQ( stride3.m_dim.N0 , 2 );
    ASSERT_EQ( stride3.m_dim.N1 , 3 );
    ASSERT_EQ( stride3.m_dim.N2 , 4 );
    ASSERT_EQ( stride3.m_dim.N3 , 1 );
    ASSERT_EQ( stride3.size() , 2 * 3 * 4 );

    int offset = 0 ;

    for ( int k = 0 ; k < 4 ; ++k ){
    for ( int j = 0 ; j < 3 ; ++j ){
    for ( int i = 0 ; i < 2 ; ++i , ++offset ){
      ASSERT_EQ( offset , dyn_off3(i,j,k) );
      ASSERT_EQ( stride3(i,j,k) , dyn_off3(i,j,k) );
    }}}

    ASSERT_EQ( dyn_off3.extent() , offset );
    ASSERT_EQ( stride3.extent() , dyn_off3.extent() );
  }

  // Large dimension is likely padded
  {
    constexpr int N0 = 2000 ;
    constexpr int N1 = 300 ;

    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s0_s0_s4 , Kokkos::LayoutLeft > left_s0_s0_s4 ;

    left_s0_s0_s4 dyn_off3( std::integral_constant<unsigned,sizeof(int)>(), N0, N1, 0, 0, 0, 0, 0, 0 );

    stride_s0_s0_s0  stride3( dyn_off3 );

    ASSERT_EQ( dyn_off3.m_dim.rank , 3 );
    ASSERT_EQ( dyn_off3.m_dim.N0 , N0 );
    ASSERT_EQ( dyn_off3.m_dim.N1 , N1 );
    ASSERT_EQ( dyn_off3.m_dim.N2 , 4 );
    ASSERT_EQ( dyn_off3.m_dim.N3 , 1 );
    ASSERT_EQ( dyn_off3.size() , N0 * N1 * 4 );

    ASSERT_EQ( stride3.m_dim.rank , 3 );
    ASSERT_EQ( stride3.m_dim.N0 , N0 );
    ASSERT_EQ( stride3.m_dim.N1 , N1 );
    ASSERT_EQ( stride3.m_dim.N2 , 4 );
    ASSERT_EQ( stride3.m_dim.N3 , 1 );
    ASSERT_EQ( stride3.size() , N0 * N1 * 4 );
    ASSERT_EQ( stride3.extent() , dyn_off3.extent() );

    int offset = 0 ;

    for ( int k = 0 ; k < 4 ; ++k ){
    for ( int j = 0 ; j < N1 ; ++j ){
    for ( int i = 0 ; i < N0 ; ++i ){
      ASSERT_LE( offset , dyn_off3(i,j,k) );
      ASSERT_EQ( stride3(i,j,k) , dyn_off3(i,j,k) );
      offset = dyn_off3(i,j,k) + 1 ;
    }}}

    ASSERT_LE( offset , dyn_off3.extent() );
  }

  //----------------------------------------
  // Static dimension
  {
    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s2_s3_s4 , Kokkos::LayoutRight > right_s2_s3_s4 ;

    ASSERT_EQ( sizeof(right_s2_s3_s4) , sizeof(dim_s2_s3_s4) );

    right_s2_s3_s4 off3 ;

    stride_s0_s0_s0  stride3( off3 );

    ASSERT_EQ( off3.stride_0() , 12 );
    ASSERT_EQ( off3.stride_1() , 4 );
    ASSERT_EQ( off3.stride_2() , 1 );

    ASSERT_EQ( off3.dimension_0() , stride3.dimension_0() );
    ASSERT_EQ( off3.dimension_1() , stride3.dimension_1() );
    ASSERT_EQ( off3.dimension_2() , stride3.dimension_2() );
    ASSERT_EQ( off3.stride_0() , stride3.stride_0() );
    ASSERT_EQ( off3.stride_1() , stride3.stride_1() );
    ASSERT_EQ( off3.stride_2() , stride3.stride_2() );
    ASSERT_EQ( off3.extent() , stride3.extent() );

    int offset = 0 ;

    for ( int i = 0 ; i < 2 ; ++i ){
    for ( int j = 0 ; j < 3 ; ++j ){
    for ( int k = 0 ; k < 4 ; ++k , ++offset ){
      ASSERT_EQ( off3(i,j,k) , offset );
      ASSERT_EQ( off3(i,j,k) , stride3(i,j,k) );
    }}}

    ASSERT_EQ( off3.extent() , offset );
  }

  //----------------------------------------
  // Small dimension is unpadded
  {
    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s0_s0_s4 , Kokkos::LayoutRight > right_s0_s0_s4 ;

    right_s0_s0_s4 dyn_off3( std::integral_constant<unsigned,sizeof(int)>(), 2, 3, 0, 0, 0, 0, 0, 0 );

    stride_s0_s0_s0  stride3( dyn_off3 );

    ASSERT_EQ( dyn_off3.m_dim.rank , 3 );
    ASSERT_EQ( dyn_off3.m_dim.N0 , 2 );
    ASSERT_EQ( dyn_off3.m_dim.N1 , 3 );
    ASSERT_EQ( dyn_off3.m_dim.N2 , 4 );
    ASSERT_EQ( dyn_off3.m_dim.N3 , 1 );
    ASSERT_EQ( dyn_off3.size() , 2 * 3 * 4 );

    ASSERT_EQ( dyn_off3.dimension_0() , stride3.dimension_0() );
    ASSERT_EQ( dyn_off3.dimension_1() , stride3.dimension_1() );
    ASSERT_EQ( dyn_off3.dimension_2() , stride3.dimension_2() );
    ASSERT_EQ( dyn_off3.stride_0() , stride3.stride_0() );
    ASSERT_EQ( dyn_off3.stride_1() , stride3.stride_1() );
    ASSERT_EQ( dyn_off3.stride_2() , stride3.stride_2() );
    ASSERT_EQ( dyn_off3.extent() , stride3.extent() );

    int offset = 0 ;

    for ( int i = 0 ; i < 2 ; ++i ){
    for ( int j = 0 ; j < 3 ; ++j ){
    for ( int k = 0 ; k < 4 ; ++k , ++offset ){
      ASSERT_EQ( offset , dyn_off3(i,j,k) );
      ASSERT_EQ( dyn_off3(i,j,k) , stride3(i,j,k) );
    }}}

    ASSERT_EQ( dyn_off3.extent() , offset );
  }

  // Large dimension is likely padded
  {
    constexpr int N0 = 2000 ;
    constexpr int N1 = 300 ;

    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s0_s0_s4 , Kokkos::LayoutRight > right_s0_s0_s4 ;

    right_s0_s0_s4 dyn_off3( std::integral_constant<unsigned,sizeof(int)>(), N0, N1, 0, 0, 0, 0, 0, 0 );

    stride_s0_s0_s0  stride3( dyn_off3 );

    ASSERT_EQ( dyn_off3.m_dim.rank , 3 );
    ASSERT_EQ( dyn_off3.m_dim.N0 , N0 );
    ASSERT_EQ( dyn_off3.m_dim.N1 , N1 );
    ASSERT_EQ( dyn_off3.m_dim.N2 , 4 );
    ASSERT_EQ( dyn_off3.m_dim.N3 , 1 );
    ASSERT_EQ( dyn_off3.size() , N0 * N1 * 4 );

    ASSERT_EQ( dyn_off3.dimension_0() , stride3.dimension_0() );
    ASSERT_EQ( dyn_off3.dimension_1() , stride3.dimension_1() );
    ASSERT_EQ( dyn_off3.dimension_2() , stride3.dimension_2() );
    ASSERT_EQ( dyn_off3.stride_0() , stride3.stride_0() );
    ASSERT_EQ( dyn_off3.stride_1() , stride3.stride_1() );
    ASSERT_EQ( dyn_off3.stride_2() , stride3.stride_2() );
    ASSERT_EQ( dyn_off3.extent() , stride3.extent() );

    int offset = 0 ;

    for ( int i = 0 ; i < N0 ; ++i ){
    for ( int j = 0 ; j < N1 ; ++j ){
    for ( int k = 0 ; k < 4 ; ++k ){
      ASSERT_LE( offset , dyn_off3(i,j,k) );
      ASSERT_EQ( dyn_off3(i,j,k) , stride3(i,j,k) );
      offset = dyn_off3(i,j,k) + 1 ;
    }}}

    ASSERT_LE( offset , dyn_off3.extent() );
  }

  //----------------------------------------
  // Subview
  {
    constexpr int N0 = 2000 ;
    constexpr int N1 = 300 ;

    constexpr int sub_N0 = 1000 ;
    constexpr int sub_N1 = 200 ;
    constexpr int sub_N2 = 4 ;

    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s0_s0_s4 , Kokkos::LayoutLeft > left_s0_s0_s4 ;

    left_s0_s0_s4 dyn_off3( std::integral_constant<unsigned,sizeof(int)>(), N0, N1, 0, 0, 0, 0, 0, 0 );

    stride_s0_s0_s0  stride3( dyn_off3 , sub_N0 , sub_N1 , sub_N2 , 0 , 0 , 0 , 0 , 0 );

    ASSERT_EQ( stride3.dimension_0() , sub_N0 );
    ASSERT_EQ( stride3.dimension_1() , sub_N1 );
    ASSERT_EQ( stride3.dimension_2() , sub_N2 );
    ASSERT_EQ( stride3.size() , sub_N0 * sub_N1 * sub_N2 );

    ASSERT_EQ( dyn_off3.stride_0() , stride3.stride_0() );
    ASSERT_EQ( dyn_off3.stride_1() , stride3.stride_1() );
    ASSERT_EQ( dyn_off3.stride_2() , stride3.stride_2() );
    ASSERT_GE( dyn_off3.extent()   , stride3.extent() );

    for ( int k = 0 ; k < sub_N2 ; ++k ){
    for ( int j = 0 ; j < sub_N1 ; ++j ){
    for ( int i = 0 ; i < sub_N0 ; ++i ){
      ASSERT_EQ( stride3(i,j,k) , dyn_off3(i,j,k) );
    }}}
  }

  {
    constexpr int N0 = 2000 ;
    constexpr int N1 = 300 ;

    constexpr int sub_N0 = 1000 ;
    constexpr int sub_N1 = 200 ;
    constexpr int sub_N2 = 4 ;

    typedef Kokkos::Experimental::Impl::ViewOffset< dim_s0_s0_s4 , Kokkos::LayoutRight > right_s0_s0_s4 ;

    right_s0_s0_s4 dyn_off3( std::integral_constant<unsigned,sizeof(int)>(), N0, N1, 0, 0, 0, 0, 0, 0 );

    stride_s0_s0_s0  stride3( dyn_off3 , sub_N0 , sub_N1 , sub_N2 , 0 , 0 , 0 , 0 , 0 );

    ASSERT_EQ( stride3.dimension_0() , sub_N0 );
    ASSERT_EQ( stride3.dimension_1() , sub_N1 );
    ASSERT_EQ( stride3.dimension_2() , sub_N2 );
    ASSERT_EQ( stride3.size() , sub_N0 * sub_N1 * sub_N2 );

    ASSERT_EQ( dyn_off3.stride_0() , stride3.stride_0() );
    ASSERT_EQ( dyn_off3.stride_1() , stride3.stride_1() );
    ASSERT_EQ( dyn_off3.stride_2() , stride3.stride_2() );
    ASSERT_GE( dyn_off3.extent()   , stride3.extent() );

    for ( int i = 0 ; i < sub_N0 ; ++i ){
    for ( int j = 0 ; j < sub_N1 ; ++j ){
    for ( int k = 0 ; k < sub_N2 ; ++k ){
      ASSERT_EQ( stride3(i,j,k) , dyn_off3(i,j,k) );
    }}}
  }

  //----------------------------------------
  {
    constexpr int N = 1000 ;

    test_view_range( N , N / 2 , N / 2 , 0 );
    test_view_range( N , Kokkos::ALL() , 0 , N );
    test_view_range( N , std::pair<int,int>( N / 4 , 10 + N / 4 ) , N / 4 , 10 );
    test_view_range( N , Kokkos::pair<int,int>( N / 4 , 10 + N / 4 ) , N / 4 , 10 );
  }
  //----------------------------------------
  // view data analysis

  {
    using a_const_int_r1 = Kokkos::Experimental::Impl::ViewDataAnalysis< const int[] > ;

    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::specialize , void >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::dimension , Kokkos::Experimental::Impl::ViewDimension<0> >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::type , const int[] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::value_type , const int >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::array_scalar_type , const int[] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::const_type , const int[] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::const_value_type , const int >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::const_array_scalar_type , const int[] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::non_const_type , int [] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r1::non_const_value_type , int >::value ));

    using a_const_int_r3 = Kokkos::Experimental::Impl::ViewDataAnalysis< const int**[4] > ;

    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::specialize , void >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::dimension , Kokkos::Experimental::Impl::ViewDimension<0,0,4> >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::type , const int**[4] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::value_type , const int >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::array_scalar_type , const int**[4] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::const_type , const int**[4] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::const_value_type , const int >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::const_array_scalar_type , const int**[4] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::non_const_type , int**[4] >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::non_const_value_type , int >::value ));
    ASSERT_TRUE( ( std::is_same< typename a_const_int_r3::non_const_array_scalar_type , int**[4] >::value ));
  }

  //----------------------------------------

  {
    constexpr int N = 10 ;

    using T = Kokkos::Experimental::View<int*,HostExecSpace> ;
    using C = Kokkos::Experimental::View<const int*,HostExecSpace> ;

    int data[N] ;

    T vr1(data,N);
    C cr1(vr1);

    // Generate static_assert error:
    // T tmp( cr1 );

    ASSERT_EQ( vr1.extent() , N );
    ASSERT_EQ( cr1.extent() , N );
    ASSERT_EQ( vr1.data() , & data[0] );
    ASSERT_EQ( cr1.data() , & data[0] );

    ASSERT_TRUE( ( std::is_same< typename T::data_type           , int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::const_data_type     , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::non_const_data_type , int* >::value ) );

    ASSERT_TRUE( ( std::is_same< typename T::array_scalar_type           , int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::const_array_scalar_type     , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::non_const_array_scalar_type , int* >::value ) );

    ASSERT_TRUE( ( std::is_same< typename T::value_type           , int >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::const_value_type     , const int >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::non_const_value_type , int >::value ) );

    ASSERT_TRUE( ( std::is_same< typename T::memory_space , typename HostExecSpace::memory_space >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::reference_type , int & >::value ) );

    ASSERT_EQ( T::Rank , 1 );

    ASSERT_TRUE( ( std::is_same< typename C::data_type           , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename C::const_data_type     , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename C::non_const_data_type , int* >::value ) );

    ASSERT_TRUE( ( std::is_same< typename C::array_scalar_type           , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename C::const_array_scalar_type     , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename C::non_const_array_scalar_type , int* >::value ) );

    ASSERT_TRUE( ( std::is_same< typename C::value_type           , const int >::value ) );
    ASSERT_TRUE( ( std::is_same< typename C::const_value_type     , const int >::value ) );
    ASSERT_TRUE( ( std::is_same< typename C::non_const_value_type , int >::value ) );

    ASSERT_TRUE( ( std::is_same< typename C::memory_space , typename HostExecSpace::memory_space >::value ) );
    ASSERT_TRUE( ( std::is_same< typename C::reference_type , const int & >::value ) );

    ASSERT_EQ( C::Rank , 1 );

    ASSERT_EQ( vr1.dimension_0() , N );

    for ( int i = 0 ; i < N ; ++i ) data[i] = i + 1 ;
    for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( vr1[i] , i + 1 );
    for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( cr1[i] , i + 1 );

    {
      T tmp( vr1 );
      for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( tmp[i] , i + 1 );
      for ( int i = 0 ; i < N ; ++i ) vr1(i) = i + 2 ;
      for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( tmp[i] , i + 2 );
    }

    for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( vr1[i] , i + 2 );
  }

  {
    constexpr int N = 10 ;
    using T = Kokkos::Experimental::View<int*,HostExecSpace> ;
    using C = Kokkos::Experimental::View<const int*,HostExecSpace> ;

    T vr1("vr1",N);
    C cr1(vr1);

    ASSERT_TRUE( ( std::is_same< typename T::data_type           , int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::const_data_type     , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::non_const_data_type , int* >::value ) );

    ASSERT_TRUE( ( std::is_same< typename T::array_scalar_type           , int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::const_array_scalar_type     , const int* >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::non_const_array_scalar_type , int* >::value ) );

    ASSERT_TRUE( ( std::is_same< typename T::value_type           , int >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::const_value_type     , const int >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::non_const_value_type , int >::value ) );

    ASSERT_TRUE( ( std::is_same< typename T::memory_space , typename HostExecSpace::memory_space >::value ) );
    ASSERT_TRUE( ( std::is_same< typename T::reference_type , int & >::value ) );
    ASSERT_EQ( T::Rank , 1 );
 
    ASSERT_EQ( vr1.dimension_0() , N );

    for ( int i = 0 ; i < N ; ++i ) vr1(i) = i + 1 ;
    for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( vr1[i] , i + 1 );
    for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( cr1[i] , i + 1 );

    {
      T tmp( vr1 );
      for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( tmp[i] , i + 1 );
      for ( int i = 0 ; i < N ; ++i ) vr1(i) = i + 2 ;
      for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( tmp[i] , i + 2 );
    }

    for ( int i = 0 ; i < N ; ++i ) ASSERT_EQ( vr1[i] , i + 2 );
  }
}

template< class HostExecSpace >
void test_view_mapping_subview()
{
  {
    constexpr int N = 10 ;
    using T = Kokkos::Experimental::View<int*,HostExecSpace> ;
    using S = Kokkos::Experimental::Subview< T , true > ;

    T a("a",N);
    S b = Kokkos::Experimental::subview( a , std::pair<int,int>(1,N-1) );

    for ( int i = 1 ; i < N-1 ; ++i ) ASSERT_EQ( & a[i] , & b[i-1] );
  }

  {
    constexpr int N0 = 10 , N1 = 11 , N2 = 12 ;
    using T = Kokkos::Experimental::View<int***,HostExecSpace> ;
    using S = Kokkos::Experimental::Subview< T , true , true , true > ;

    T a("a",N0,N1,N2);
    S b = Kokkos::Experimental::subview( a
                                       , std::pair<int,int>(1,N0-1)
                                       , std::pair<int,int>(1,N1-1)
                                       , std::pair<int,int>(1,N2-1)
                                       );

    for ( int i2 = 1 ; i2 < N2-1 ; ++i2 ) {
    for ( int i1 = 1 ; i1 < N1-1 ; ++i1 ) {
    for ( int i0 = 1 ; i0 < N0-1 ; ++i0 ) {
      ASSERT_EQ( & a(i0,i1,i2) , & b(i0-1,i1-1,i2-1) );
    }}}
  }

  {
    constexpr int N0 = 10 , N1 = 11 , N2 = 12 ;
    using T = Kokkos::Experimental::View<int***[13][14],HostExecSpace> ;
    using S = Kokkos::Experimental::Subview< T , true , true , true , false , false > ;

    T a("a",N0,N1,N2);
    S b = Kokkos::Experimental::subview( a
                                       , std::pair<int,int>(1,N0-1)
                                       , std::pair<int,int>(1,N1-1)
                                       , std::pair<int,int>(1,N2-1)
                                       , 1
                                       , 2
                                       );

    for ( int i2 = 1 ; i2 < N2-1 ; ++i2 ) {
    for ( int i1 = 1 ; i1 < N1-1 ; ++i1 ) {
    for ( int i0 = 1 ; i0 < N0-1 ; ++i0 ) {
      ASSERT_EQ( & a(i0,i1,i2,1,2) , & b(i0-1,i1-1,i2-1) );
    }}}
  }

  {
    constexpr int N0 = 10 , N1 = 11 , N2 = 12 ;
    using T = Kokkos::Experimental::View<int***[13][14],HostExecSpace> ;
    using S = Kokkos::Experimental::Subview< T , false , true , true , true , false > ;

    T a("a",N0,N1,N2);
    S b = Kokkos::Experimental::subview( a
                                       , 1
                                       , std::pair<int,int>(1,N0-1)
                                       , std::pair<int,int>(1,N1-1)
                                       , std::pair<int,int>(1,N2-1)
                                       , 2
                                       );

    ASSERT_EQ( a.stride_1() , b.stride_0() );
    ASSERT_EQ( a.stride_2() , b.stride_1() );
    ASSERT_EQ( a.stride_3() , b.stride_2() );

    for ( int i2 = 1 ; i2 < N2-1 ; ++i2 ) {
    for ( int i1 = 1 ; i1 < N1-1 ; ++i1 ) {
    for ( int i0 = 1 ; i0 < N0-1 ; ++i0 ) {
      ASSERT_EQ( & a(1,i0,i1,i2,2) , & b(i0-1,i1-1,i2-1) );
    }}}
  }
}

/*--------------------------------------------------------------------------*/

template< class ViewType >
struct TestViewMapOperator {

  static_assert( ViewType::reference_type_is_lvalue
               , "Test only valid for lvalue reference type" );

  const ViewType v ;

  KOKKOS_INLINE_FUNCTION
  void test_left( size_t i0 , long & error_count ) const
    {
      typename ViewType::value_type * const base_ptr = & v(0,0,0,0,0,0,0,0);
      const size_t n1 = v.dimension_1();
      const size_t n2 = v.dimension_2();
      const size_t n3 = v.dimension_3();
      const size_t n4 = v.dimension_4();
      const size_t n5 = v.dimension_5();
      const size_t n6 = v.dimension_6();
      const size_t n7 = v.dimension_7();

      size_t offset = 0 ;

      for ( size_t i7 = 0 ; i7 < n7 ; ++i7 )
      for ( size_t i6 = 0 ; i6 < n6 ; ++i6 )
      for ( size_t i5 = 0 ; i5 < n5 ; ++i5 )
      for ( size_t i4 = 0 ; i4 < n4 ; ++i4 )
      for ( size_t i3 = 0 ; i3 < n3 ; ++i3 )
      for ( size_t i2 = 0 ; i2 < n2 ; ++i2 )
      for ( size_t i1 = 0 ; i1 < n1 ; ++i1 )
      {
        const long d = & v(i0,i1,i2,i3,i4,i5,i6,i7) - base_ptr ;
        if ( d < offset ) ++error_count ;
        offset = d ;
      }

      if ( v.extent() <= offset ) ++error_count ;
    }

  KOKKOS_INLINE_FUNCTION
  void test_right( size_t i0 , long & error_count ) const
    {
      typename ViewType::value_type * const base_ptr = & v(0,0,0,0,0,0,0,0);
      const size_t n1 = v.dimension_1();
      const size_t n2 = v.dimension_2();
      const size_t n3 = v.dimension_3();
      const size_t n4 = v.dimension_4();
      const size_t n5 = v.dimension_5();
      const size_t n6 = v.dimension_6();
      const size_t n7 = v.dimension_7();

      size_t offset = 0 ;

      for ( size_t i1 = 0 ; i1 < n1 ; ++i1 )
      for ( size_t i2 = 0 ; i2 < n2 ; ++i2 )
      for ( size_t i3 = 0 ; i3 < n3 ; ++i3 )
      for ( size_t i4 = 0 ; i4 < n4 ; ++i4 )
      for ( size_t i5 = 0 ; i5 < n5 ; ++i5 )
      for ( size_t i6 = 0 ; i6 < n6 ; ++i6 )
      for ( size_t i7 = 0 ; i7 < n7 ; ++i7 )
      {
        const long d = & v(i0,i1,i2,i3,i4,i5,i6,i7) - base_ptr ;
        if ( d < offset ) ++error_count ;
        offset = d ;
      }

      if ( v.extent() <= offset ) ++error_count ;
    }

  KOKKOS_INLINE_FUNCTION
  void operator()( size_t i , long & error_count ) const
    {
      if ( std::is_same< typename ViewType::array_layout , Kokkos::LayoutLeft >::value )
        test_left(i,error_count);
      else if ( std::is_same< typename ViewType::array_layout , Kokkos::LayoutRight >::value )
        test_right(i,error_count);
    }

  TestViewMapOperator() : v( "Test" , 10 , 9 , 8 , 7 , 6 , 5 , 4 , 3 ) { }

  static void run()
    {
      TestViewMapOperator self ;
      long error_count ;
      Kokkos::RangePolicy< typename ViewType::execution_space > range(0,self.v.dimension_0());
      Kokkos::parallel_reduce( range , self , error_count );
      ASSERT_EQ( 0 , error_count );
    }
};


template< class ExecSpace >
void test_view_mapping_operator()
{
  TestViewMapOperator< Kokkos::Experimental::View<int,Kokkos::LayoutLeft,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int*,Kokkos::LayoutLeft,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int**,Kokkos::LayoutLeft,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int***,Kokkos::LayoutLeft,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int****,Kokkos::LayoutLeft,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int*****,Kokkos::LayoutLeft,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int******,Kokkos::LayoutLeft,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int*******,Kokkos::LayoutLeft,ExecSpace> >::run();

  TestViewMapOperator< Kokkos::Experimental::View<int,Kokkos::LayoutRight,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int*,Kokkos::LayoutRight,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int**,Kokkos::LayoutRight,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int***,Kokkos::LayoutRight,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int****,Kokkos::LayoutRight,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int*****,Kokkos::LayoutRight,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int******,Kokkos::LayoutRight,ExecSpace> >::run();
  TestViewMapOperator< Kokkos::Experimental::View<int*******,Kokkos::LayoutRight,ExecSpace> >::run();
}



} /* namespace Test */

/*--------------------------------------------------------------------------*/

