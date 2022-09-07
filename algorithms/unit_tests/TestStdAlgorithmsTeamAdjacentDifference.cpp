/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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
*/

#include <TestStdAlgorithmsCommon.hpp>
#include <cstddef>
#include <functional>
#include <numeric>
#include "gtest/gtest.h"
#include "std_algorithms/Kokkos_BeginEnd.hpp"

namespace Test {
namespace stdalgos {
namespace TestAdjacentDifference {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct PlusFunctor {
  KOKKOS_INLINE_FUNCTION constexpr ValueType operator()(
      const ValueType& lhs, const ValueType& rhs) const {
    return lhs + rhs;
  }
};

template <class ViewType, class DiffsViewType, class BinaryOp>
struct TestFunctorA {
  ViewType m_view;
  DiffsViewType m_diffsView;
  int m_apiPick;
  BinaryOp m_binaryOp;

  TestFunctorA(const ViewType view, const DiffsViewType diffsView, int apiPick,
               BinaryOp binaryOp)
      : m_view(view),
        m_diffsView(diffsView),
        m_apiPick(apiPick),
        m_binaryOp(std::move(binaryOp)) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowView  = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());
    auto myDiffView = Kokkos::subview(m_diffsView, myRowIndex, Kokkos::ALL());

    switch (m_apiPick) {
      case 0: {
        KE::adjacent_difference(member, KE::begin(myRowView),
                                KE::end(myRowView), KE::begin(myDiffView));
        break;
      }

      case 1: {
        KE::adjacent_difference(member, KE::begin(myRowView),
                                KE::end(myRowView), KE::begin(myDiffView),
                                m_binaryOp);
        break;
      }

      case 2: {
        KE::adjacent_difference(member, myRowView, myDiffView);
        break;
      }

      case 3: {
        KE::adjacent_difference(member, myRowView, myDiffView, m_binaryOp);
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level adjacent_difference
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  auto [dataView, cloneOfDataViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          Kokkos::pair{ValueType(5), ValueType(523)}, "dataView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // to verify that things work, each team stores the result
  // of its adjacent_difference call, and then we check
  // that these match what we expect
  Kokkos::View<ValueType**> diffsView("diffsView", numTeams, numCols);

  // use CTAD for functor
  TestFunctorA fnc(dataView, diffsView, apiId, PlusFunctor<ValueType>{});
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto dataView_h = create_mirror_view(Kokkos::HostSpace(), dataView);
  Kokkos::deep_copy(dataView_h, dataView);
  auto diffsView_h = create_mirror_view(Kokkos::HostSpace(), diffsView);
  Kokkos::deep_copy(diffsView_h, diffsView);

  Kokkos::View<ValueType**> diffsView_gold_h("diffsView_gold_h", numTeams,
                                             numCols);

  for (std::size_t i = 0; i < diffsView_h.extent(0); ++i) {
    auto dataViewRow_h = Kokkos::subview(dataView_h, i, Kokkos::ALL());
    auto diffsViewRow_gold_h =
        Kokkos::subview(diffsView_gold_h, i, Kokkos::ALL());

    switch (apiId) {
      case 0:
      case 2: {
        std::adjacent_difference(KE::begin(dataViewRow_h),
                                 KE::end(dataViewRow_h),
                                 KE::begin(diffsViewRow_gold_h));
        break;
      }

      case 1:
      case 3: {
        std::adjacent_difference(
            KE::begin(dataViewRow_h), KE::end(dataViewRow_h),
            KE::begin(diffsViewRow_gold_h), PlusFunctor<ValueType>{});
      }
    }

    auto diffsViewRow_h = Kokkos::subview(diffsView_h, i, Kokkos::ALL());
    for (std::size_t j = 0; j < diffsViewRow_gold_h.extent(0); ++j) {
      EXPECT_FLOAT_EQ(diffsViewRow_gold_h(j), diffsViewRow_h(j));
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_adjacent_difference_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TestAdjacentDifference
}  // namespace stdalgos
}  // namespace Test
