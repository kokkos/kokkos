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
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER

#ifndef CONTAINERS_UNIT_TESTS_TESTOFFSETVIEW_HPP_
#define CONTAINERS_UNIT_TESTS_TESTOFFSETVIEW_HPP_



#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_OffsetView.hpp>

using std::endl;
using std::cout;

namespace Test{


  template <typename Scalar, typename Device>
  void test_offsetview_construction(unsigned int size)
  {

    typedef Kokkos::OffsetView<Scalar**, Device> offset_view_type;
    typedef Kokkos::View<Scalar**, Device> view_type;

    Kokkos::index_list_type range0 = {-1, 3};
    Kokkos::index_list_type range1 = {-2, 2};

    offset_view_type ov("firstOV", range0, range1);

    ASSERT_EQ("firstOV", ov.label());
    ASSERT_EQ(2, ov.Rank);

    ASSERT_EQ(ov.begin(0), -1);
    ASSERT_EQ(ov.end(0), 4);

    ASSERT_EQ(ov.begin(1), -2);
    ASSERT_EQ(ov.end(1), 3);

    ASSERT_EQ(ov.extent(0), 5);
    ASSERT_EQ(ov.extent(1), 5);

    const int ovmin0 = ov.begin(0);
    const int ovend0 = ov.end(0);
    const int ovmin1 = ov.begin(1);
    const int ovend1 = ov.end(1);

    auto x = ovmin0;

    for(int i = ovmin0; i < ovend0; ++i) {
      for(int j = ovmin1 ; j < ovend1; ++j) {
        ov(i,j) = i + j;
      }
    }

    for(int i = ovmin0; i < ovend0; ++i) {
      for(int j = ovmin1 ; j < ovend1; ++j) {
        ASSERT_EQ(ov(i,j),  i + j) << "Bad data found in View";
      }
    }

    {
      offset_view_type ovCopy(ov);
      ASSERT_EQ(ovCopy==ov, true) << "Copy constructor or equivalence operator broken";
    }
    {
      offset_view_type ovAssigned = ov;
      ASSERT_EQ(ovAssigned==ov, true) <<  "Assignment operator or equivalence operator broken";
    }

    auto viewFromOV = ov.view();
    ASSERT_EQ(viewFromOV == ov, true) << "Assignment of OffsetView to View or equivalence operator View == OffsetView broken";

    {
      offset_view_type ovFromV(viewFromOV, {-1, -2});
      ASSERT_EQ(ovFromV == viewFromOV , true) << "Construction of OffsetView from View or equivalence operator OffsetView == View broken";
    }
  }


  TEST_F( TEST_CATEGORY, offsetview_construction) {
      test_offsetview_construction<int,TEST_EXECSPACE>(10);
  }

} // namespace Test

#endif /* CONTAINERS_UNIT_TESTS_TESTOFFSETVIEW_HPP_ */
