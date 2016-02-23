
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

#ifndef KOKKOS_TEST_DYNRANKVIEW_HPP
#define KOKKOS_TEST_DYNRANKVIEW_HPP

#include <gtest/gtest.h>
#include <iostream>

namespace Test {

template< typename DataType , class Device >
struct test_dynamic_rank_view {

  typedef typename Kokkos::DynRankView< DataType, Device > dynview;
  dynview m_testview;

  //set and check values:
  void set_values( size_t s0 , size_t s1 , size_t s2 , size_t s3 , size_t s4 , size_t s5 , size_t s6 , size_t s7 ) {
    for ( size_t i0 = 0; i0 < s0; ++i0 ) {
    for ( size_t i1 = 0; i1 < s1; ++i1 ) {
    for ( size_t i2 = 0; i2 < s2; ++i2 ) {
    for ( size_t i3 = 0; i3 < s3; ++i3 ) {
    for ( size_t i4 = 0; i4 < s4; ++i4 ) {
    for ( size_t i5 = 0; i5 < s5; ++i5 ) {
    for ( size_t i6 = 0; i6 < s6; ++i6 ) {
    for ( size_t i7 = 0; i7 < s7; ++i7 ) {
      m_testview(i0 , i1 , i2 , i3 , i4 , i5 , i6 , i7) = i7*s0*s1*s2*s3*s4*s5*s6 + i6*s0*s1*s2*s3*s4*s5 + i5*s0*s1*s2*s3*s4 + i4*s0*s1*s2*s3 + i3*s0*s1*s2 + i2*s0*s1 + i1*s0 + i0;
    }}}}}}}}

  }

  void check_values( size_t s0 , size_t s1 , size_t s2 , size_t s3 , size_t s4 , size_t s5 , size_t s6 , size_t s7 ) {
    size_t error_count = 0;

    for ( size_t i0 = 0; i0 < s0; ++i0 ) {
    for ( size_t i1 = 0; i1 < s1; ++i1 ) {
    for ( size_t i2 = 0; i2 < s2; ++i2 ) {
    for ( size_t i3 = 0; i3 < s3; ++i3 ) {
    for ( size_t i4 = 0; i4 < s4; ++i4 ) {
    for ( size_t i5 = 0; i5 < s5; ++i5 ) {
    for ( size_t i6 = 0; i6 < s6; ++i6 ) {
    for ( size_t i7 = 0; i7 < s7; ++i7 ) {
      if ( m_testview(i0 , i1 , i2 , i3 , i4 , i5 , i6 , i7) != i7*s0*s1*s2*s3*s4*s5*s6 + i6*s0*s1*s2*s3*s4*s5 + i5*s0*s1*s2*s3*s4 + i4*s0*s1*s2*s3 + i3*s0*s1*s2 + i2*s0*s1 + i1*s0 + i0 )
        { ++error_count; }
    }}}}}}}}

    ASSERT_EQ( error_count , size_t(0) ); //should be 0, intentionally wrong to make sure test is running
  }

  //constructors
  //test_dynamic_rank_view(const size_t s0 = 0, const size_t s1 = 0, const size_t s2 = 0, const size_t s3 = 0, const size_t s4 = 0, const size_t s5 = 0, const size_t s6 = 0, const size_t s7 = 0) : m_testview(s0 , s1 , s2 , s3 , s4 , s5 , s6 , s7)
  //
//  test_dynamic_rank_view(size_t s0 = 0, size_t s1 = 0, size_t s2 = 0, size_t s3 = 0, size_t s4 = 0, size_t s5 = 0, size_t s6 = 0, size_t s7 = 0) : m_testview(s0 , s1 , s2 , s3 , s4 , s5 , s6 , s7)
  test_dynamic_rank_view(size_t s0 = 1, size_t s1 = 1, size_t s2 = 1, size_t s3 = 1, size_t s4 = 1, size_t s5 = 1, size_t s6 = 1, size_t s7 = 1) : m_testview(s0 , s1 , s2 , s3 , s4 , s5 , s6 , s7)
  {
    set_values(s0 , s1 , s2 , s3 , s4 , s5 , s6 , s7);
    check_values(s0 , s1 , s2 , s3 , s4 , s5 , s6 , s7);
    std::cout << "Rank of this view is " <<  m_testview.Rank << std::endl;
    std::cout << "extent of dim0 is " << m_testview.extent(0) << std::endl;
    std::cout << "extent_int of dim1 is " << m_testview.extent_int(1) << std::endl;
    std::cout << "size() is " << m_testview.size() << std::endl;
  }

};

} //end Test


#endif
