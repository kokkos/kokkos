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
#include <algorithm>

namespace Test {
namespace stdalgos {
namespace TeamRemove {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist(int b, std::size_t seedIn) : m_dist(0, b) { m_gen.seed(seedIn); }

  int operator()() { return m_dist(m_gen); }
};

template <class ViewType, class DistancesViewType, class ValueType>
struct TestFunctorA {
  ViewType m_view;
  ValueType m_targetValue;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const ViewType view, ValueType oldVal,
               const DistancesViewType distancesView, int apiPick)
      : m_view(view),
        m_targetValue(oldVal),
        m_distancesView(distancesView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    if (m_apiPick == 0) {
      auto it = KE::remove(member, KE::begin(myRowView), KE::end(myRowView),
                           m_targetValue);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::remove(member, myRowView, m_targetValue);
      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     set a random subset of each row of a rank-2 view
     to a target value and run a team-level KE::remove
     with one team per row to remove all those elements.
   */

  const auto targetVal = static_cast<ValueType>(531);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // Create a view in the memory space associated with default exespace with as
  // many rows as the number of teams and fill it with random values from an
  // arbitrary range. Pick range so that some of the values are equal to target.
  auto [dataView, dataView_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols,
      Kokkos::pair<ValueType, ValueType>{targetVal - 1, targetVal + 1},
      "dataView");

  // note that we need to count how many elements are equal
  // to targetVal because the dataView was origianlly filled
  // with random values
  std::vector<std::size_t> perRowRealCount(numTeams);
  for (std::size_t i = 0; i < dataView_h.extent(0); ++i) {
    std::size_t realCount = 0;
    for (std::size_t j = 0; j < dataView_h.extent(1); ++j) {
      if (dataView_h(i, j) == targetVal) {
        realCount++;
      }
    }
    perRowRealCount[i] = realCount;
  }

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, targetVal, distancesView, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check against std
  // -----------------------------------------------
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  auto distancesView_h   = create_host_space_copy(distancesView);
  for (std::size_t i = 0; i < dataViewAfterOp_h.extent(0); ++i) {
    auto myRow = Kokkos::subview(dataView_h, i, Kokkos::ALL());
    auto stdIt = std::remove(KE::begin(myRow), KE::end(myRow), targetVal);
    const std::size_t stdDistance = KE::distance(KE::begin(myRow), stdIt);
    ASSERT_EQ(distancesView_h(i), stdDistance);
    ASSERT_EQ(distancesView_h(i), numCols - perRowRealCount[i]);

    for (std::size_t j = 0; j < distancesView_h(i); ++j) {
      ASSERT_EQ(dataViewAfterOp_h(i, j), dataView_h(i, j));
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8113}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_remove_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamRemove
}  // namespace stdalgos
}  // namespace Test