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
#include <KokkosExp_MDRangePolicy.hpp>

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

//#ifdef KOKKOS_ENABLE_CUDA_LAMBDA
    {
      Kokkos::OffsetView<Scalar*, Device> offsetV1("OneDOffsetView", range0);

      Kokkos::RangePolicy<Device, int> rangePolicy1(range0.begin()[0], range0.begin()[1]);
      Kokkos::parallel_for(rangePolicy1, KOKKOS_LAMBDA (const int i){
        offsetV1(i) = 1;
      }
      );

      const int maxit = offsetV1.end(0);

      int OVResult = 0;
      Kokkos::parallel_reduce(maxit, KOKKOS_LAMBDA(const int i, int & updateMe){
        updateMe += offsetV1(i);
      }, OVResult);

      ASSERT_NE(OVResult, range0.begin()[1] - range0.begin()[0]) << "found wrong number of elements in OffsetView that was summed.";

//      const auto VResult = sum1(offsetV1.view());
//      ASSERT_EQ(VResult, range0.begin()[1] - range0.begin()[0]) << "found wrong number of elements in View that was summed.";

    }
//#endif

//#ifdef KOKKOS_ENABLE_CUDA_LAMBDA


    typedef Kokkos::MDRangePolicy<Device, Kokkos::Rank<2>, Kokkos::IndexType<int> > range_type;
    typedef typename range_type::point_type point_type;

    range_type rangePolicy2D(point_type{ {ovmin0, ovmin1 } },
                             point_type{ { ovend0, ovend1 } });

    Kokkos::parallel_for(rangePolicy2D, KOKKOS_LAMBDA (const int i, const int j) {
        ov(i,j) = i + j;
      }
    );
    
    typename offset_view_type::HostMirror hostView =
      Kokkos::create_mirror_view(ov);


    //need hostmirror and deep copy for this to work.
        for(int i = hostView.begin(0); i < hostView.end(0); ++i) {
          for(int j = hostView.begin(1); j < hostView.end(1); ++j) {
        	 ASSERT_EQ(hostView(i,j),  i + j) << "Bad data found in OffsetView";
          }
        }

       // Kokkos::parallel_for(rangePolicy2D, KOKKOS_LAMBDA (const int i, const int j) {
       // 	   ASSERT_EQ(ov(i,j),  i + j);
       //   }
       // );

    int OVResult = 0;
    Kokkos::parallel_reduce(rangePolicy2D, KOKKOS_LAMBDA(const int i, const int j, int & updateMe){
      updateMe += ov(i, j);
    }, OVResult);

    int answer = 0;
    for(int i = ov.begin(0); i < ov.end(0); ++i) {
      for(int j = ov.begin(1); j < ov.end(1); ++j) {
         answer += i + j;
      }
    }

    ASSERT_EQ(OVResult, answer) << "Bad data found in OffsetView";

//#endif
    {
      offset_view_type ovCopy(ov);
      ASSERT_EQ(ovCopy==ov, true) <<
          "Copy constructor or equivalence operator broken";
    }

    {
      offset_view_type ovAssigned = ov;
      ASSERT_EQ(ovAssigned==ov, true) <<
          "Assignment operator or equivalence operator broken";
    }

    view_type viewFromOV = ov.view();

    ASSERT_EQ(viewFromOV == ov, true) <<
        "OffsetView::view() or equivalence operator View == OffsetView broken";

    {
      offset_view_type ovFromV(viewFromOV, {-1, -2});

      ASSERT_EQ(ovFromV == viewFromOV , true) <<
          "Construction of OffsetView from View or equivalence operator OffsetView == View broken";
    }
    {
      offset_view_type ovFromV = viewFromOV;
      ASSERT_EQ(ovFromV == viewFromOV , true) <<
          "Construction of OffsetView from View by assignment (implicit conversion) or equivalence operator OffsetView == View broken";
    }

#if 0
    {
      view_type viewByAssignment = ov;
      ASSERT_EQ(viewFromOV == ov, true) <<
          "Assignment of OffsetView to View or equivalence operator View == OffsetView broken";

    }
#endif
  }


  TEST_F( TEST_CATEGORY, offsetview_construction) {
      test_offsetview_construction<int,TEST_EXECSPACE>(10);
  }

} // namespace Test

#endif /* CONTAINERS_UNIT_TESTS_TESTOFFSETVIEW_HPP_ */
