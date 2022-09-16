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
// Questions? Contact Christian R. Trott crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <TestStdAlgorithmsCommon.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include "gtest/gtest.h"
#include "std_algorithms/Kokkos_BeginEnd.hpp"

namespace Test {
namespace stdalgos {
namespace TestNoneOf {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  KOKKOS_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  KOKKOS_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class DataViewType, class NoneOfResultsViewType, class UnaryOp>
struct TestFunctorA {
  DataViewType m_dataView;
  NoneOfResultsViewType m_noneOfResultsView;
  int m_apiPick;
  UnaryOp m_unaryOp;

  TestFunctorA(const DataViewType dataView,
               const NoneOfResultsViewType noneOfResultsView, int apiPick,
               UnaryOp unaryOp)
      : m_dataView(dataView),
        m_noneOfResultsView(noneOfResultsView),
        m_apiPick(apiPick),
        m_unaryOp(std::move(unaryOp)) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());

    switch (m_apiPick) {
      case 0: {
        const bool result = KE::none_of(member, KE::cbegin(myRowViewFrom),
                                        KE::cend(myRowViewFrom), m_unaryOp);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_noneOfResultsView(myRowIndex) = result; });
        break;
      }

      case 1: {
        const bool result = KE::none_of(member, myRowViewFrom, m_unaryOp);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_noneOfResultsView(myRowIndex) = result; });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level none_of
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr auto lowerBound = ValueType{5};
  constexpr auto upperBound = ValueType{523};
  Kokkos::pair bounds{lowerBound, upperBound};
  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // to verify that things work, each team stores the result of its none_of
  // call, and then we check that these match what we expect
  Kokkos::View<bool*> noneOfResultsView("noneOfResultsView", numTeams);

  GreaterThanValueFunctor unaryPred{upperBound};

  // use CTAD for functor
  TestFunctorA fnc(dataView, noneOfResultsView, apiId, unaryPred);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto noneOfResultsView_h = create_host_space_copy(noneOfResultsView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());
    const bool result =
        std::none_of(KE::cbegin(rowFrom), KE::cend(rowFrom), unaryPred);
    EXPECT_EQ(result, noneOfResultsView_h(i));
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_none_of_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TestNoneOf
}  // namespace stdalgos
}  // namespace Test
