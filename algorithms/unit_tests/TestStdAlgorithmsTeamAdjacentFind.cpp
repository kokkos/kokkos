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

namespace Test {
namespace stdalgos {
namespace TestAdjacentFind {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct IsEqualFunctor {
  KOKKOS_INLINE_FUNCTION constexpr bool operator()(const ValueType& lhs,
                                                   const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DataViewType, class DistancesViewType, class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  DistancesViewType m_distancesView;
  int m_apiPick;
  BinaryPredType m_binaryPred;

  TestFunctorA(const DataViewType dataView,
               const DistancesViewType distancesView, int apiPick,
               BinaryPredType binaryPred)
      : m_dataView(dataView),
        m_distancesView(distancesView),
        m_apiPick(apiPick),
        m_binaryPred(binaryPred) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());

    switch (m_apiPick) {
      case 0: {
        const auto it = KE::adjacent_find(member, KE::cbegin(myRowViewFrom),
                                          KE::cend(myRowViewFrom));
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });
        break;
      }

      case 1: {
        const auto it = KE::adjacent_find(member, myRowViewFrom);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });
        break;
      }

      case 2: {
        const auto it =
            KE::adjacent_find(member, KE::cbegin(myRowViewFrom),
                              KE::cend(myRowViewFrom), m_binaryPred);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });
        break;
      }

      case 3: {
        const auto it = KE::adjacent_find(member, myRowViewFrom, m_binaryPred);
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
void test_A(const bool ensureAdjacentFindCanFind, std::size_t numTeams,
            std::size_t numCols, int apiId) {
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

  // If ensureAdjacentFindCanFind == true ensure there are two consecutive equal
  // elemnts in each row

  // dataView might not deep copyable (e.g. strided layout) so to prepare it
  // correclty, we make a new view that is for sure deep copyable, modify it on
  // the host, deep copy to device and then launch a kernel to copy to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  if (ensureAdjacentFindCanFind && numCols > 1) {
    for (std::size_t i = 0; i < numTeams; ++i) {
      const auto j = numCols / 2;

      dataView_dc_h(i, j - 1) = dataView_dc_h(i, j);
    }
  }

  // copy to dataView_dc and then to dataView
  Kokkos::deep_copy(dataView_dc, dataView_dc_h);

  CopyFunctorRank2 cpFun(dataView_dc, dataView);
  Kokkos::parallel_for("copy", dataView.extent(0) * dataView.extent(1), cpFun);

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
  IsEqualFunctor<ValueType> binaryPred;
  TestFunctorA fnc(dataView, distancesView, apiId, binaryPred);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom            = Kokkos::subview(dataView_dc_h, i, Kokkos::ALL());
    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);
    const std::size_t beginEndDist = KE::distance(rowFromBegin, rowFromEnd);

    switch (apiId) {
      case 0:
      case 1: {
        const auto it = std::adjacent_find(rowFromBegin, rowFromEnd);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);
        EXPECT_EQ(stdDistance, distancesView_h(i));

        if (numCols == 1) {
          EXPECT_EQ(distancesView_h(i), beginEndDist);
        } else if (ensureAdjacentFindCanFind) {
          EXPECT_NE(distancesView_h(i), beginEndDist);
        }

        break;
      }

      case 2:
      case 3: {
        const auto it =
            std::adjacent_find(rowFromBegin, rowFromEnd, binaryPred);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        EXPECT_EQ(stdDistance, distancesView_h(i));

        if (numCols == 1) {
          EXPECT_EQ(distancesView_h(i), beginEndDist);
        } else if (ensureAdjacentFindCanFind) {
          EXPECT_NE(distancesView_h(i), beginEndDist);
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool ensureAdjacentFindCanFind) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(ensureAdjacentFindCanFind, numTeams,
                                     numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_adjacent_find_team_test,
     two_consecutive_equal_elements_exist) {
  constexpr bool ensureAdjacentFindCanFind = true;

  run_all_scenarios<DynamicTag, double>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedTwoRowsTag, int>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(ensureAdjacentFindCanFind);
}

TEST(std_algorithms_adjacent_find_team_test,
     two_consecutive_equal_elements_might_exist) {
  constexpr bool ensureAdjacentFindCanFind = false;

  run_all_scenarios<DynamicTag, double>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedTwoRowsTag, int>(ensureAdjacentFindCanFind);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(ensureAdjacentFindCanFind);
}

}  // namespace TestAdjacentFind
}  // namespace stdalgos
}  // namespace Test
