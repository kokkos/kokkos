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
namespace TestAdjacentFind {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct GreaterFunctor {
  KOKKOS_INLINE_FUNCTION constexpr ValueType operator()(
      const ValueType& lhs, const ValueType& rhs) const {
    return lhs > rhs;
  }
};

template <class DataViewType, class DistancesViewType, class BinaryOp>
struct TestFunctorA {
  DataViewType m_dataView;
  DistancesViewType m_distancesView;
  int m_apiPick;
  BinaryOp m_binaryOp;

  TestFunctorA(const DataViewType dataView,
               const DistancesViewType distancesView, int apiPick,
               BinaryOp binaryOp)
      : m_dataView(dataView),
        m_distancesView(distancesView),
        m_apiPick(apiPick),
        m_binaryOp(std::move(binaryOp)) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());

    switch (m_apiPick) {
      case 0: {
        auto it = KE::adjacent_find(member, KE::cbegin(myRowViewFrom),
                                    KE::cend(myRowViewFrom));
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });
        break;
      }

      case 1: {
        auto it = KE::adjacent_find(member, myRowViewFrom);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });
        break;
      }

      case 2: {
        auto it = KE::adjacent_find(member, KE::cbegin(myRowViewFrom),
                                    KE::cend(myRowViewFrom), m_binaryOp);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });
        break;
      }

      case 3: {
        auto it = KE::adjacent_find(member, myRowViewFrom, m_binaryOp);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level adjacent_find
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols,
      Kokkos::pair{ValueType(5), ValueType(523)}, "dataView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // adjacent_find returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the beginning
  // of the interval that team operates on and then we check that these
  // distances match the std result
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, distancesView, apiId, GreaterFunctor<ValueType>{});
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());

    switch (apiId) {
      case 0:
      case 1: {
        auto it = std::adjacent_find(KE::begin(rowFrom), KE::end(rowFrom));
        const std::size_t stdDistance = KE::distance(KE::begin(rowFrom), it);
        EXPECT_EQ(stdDistance, distancesView_h(i));
        break;
      }

      case 2:
      case 3: {
        auto it = std::adjacent_find(KE::begin(rowFrom), KE::end(rowFrom),
                                     GreaterFunctor<ValueType>{});
        const std::size_t stdDistance = KE::distance(KE::begin(rowFrom), it);
        EXPECT_EQ(stdDistance, distancesView_h(i));
      }
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

TEST(std_algorithms_adjacent_find_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TestAdjacentFind
}  // namespace stdalgos
}  // namespace Test
