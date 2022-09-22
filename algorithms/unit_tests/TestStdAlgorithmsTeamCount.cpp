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
namespace TeamCount {

namespace KE = Kokkos::Experimental;

template <class ViewType, class ValuesViewType, class CountsViewType>
struct TestFunctorA {
  ViewType m_view;
  ValuesViewType m_valuesView;
  CountsViewType m_countsView;
  int m_apiPick;

  TestFunctorA(const ViewType view, const ValuesViewType valuesView,
               const CountsViewType countsView, int apiPick)
      : m_view(view),
        m_valuesView(valuesView),
        m_countsView(countsView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto rowIndex = member.league_rank();
    const auto value    = m_valuesView(rowIndex);
    auto rowView        = Kokkos::subview(m_view, rowIndex, Kokkos::ALL());

    switch (m_apiPick) {
      case 0: {
        auto result =
            KE::count(member, KE::cbegin(rowView), KE::cend(rowView), value);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_countsView(rowIndex) = result; });

        break;
      }

      case 1: {
        auto result = KE::count(member, rowView, value);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_countsView(rowIndex) = result; });

        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool searched_value_exist, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level count
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.

  // Boundaries choosen so that every drawn number is at least once in the given
  // row
  const ValueType lowerBound          = numCols / 4;
  const ValueType upperBound          = 1 + numCols * 3 / 4;
  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, Kokkos::pair{lowerBound, upperBound},
      "dataView");

  // If searched_value_exist == true, we want to ensure that count result is >
  // 0, so we randomly pick a value to look for from a given row.
  //
  // If searched_value_exist == false, we want to ensure that count returns 0,
  // so we pick a value that's outside of view boundaries.
  Kokkos::View<ValueType*> valuesView("valuesView", numTeams);
  auto valuesView_h = create_mirror_view(Kokkos::HostSpace(), valuesView);

  using rand_pool =
      Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
  rand_pool pool(lowerBound * upperBound);

  if (searched_value_exist) {
    Kokkos::View<std::size_t*, Kokkos::DefaultHostExecutionSpace> randomIndices(
        "randomIndices", numTeams);
    Kokkos::fill_random(randomIndices, pool, 0, numCols);

    for (std::size_t i = 0; i < numTeams; ++i) {
      const std::size_t j = randomIndices(i);
      valuesView_h(i)     = dataViewBeforeOp_h(i, j);
    }
  } else {
    Kokkos::fill_random(valuesView_h, pool, upperBound, upperBound * 2);
  }

  Kokkos::deep_copy(valuesView, valuesView_h);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // to verify that things work, each team stores the result of its count
  // call, and then we check that these match what we expect
  Kokkos::View<std::size_t*> countsView("countsView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, valuesView, countsView, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto countsView_h = create_host_space_copy(countsView);
  if (searched_value_exist) {
    for (std::size_t i = 0; i < dataView.extent(0); ++i) {
      auto rowFrom = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());
      const auto rowFromBegin = KE::cbegin(rowFrom);
      const auto rowFromEnd   = KE::cend(rowFrom);
      const auto val          = valuesView_h(i);

      const std::size_t result = std::count(rowFromBegin, rowFromEnd, val);
      EXPECT_EQ(result, countsView_h(i));
    }
  } else {
    for (std::size_t i = 0; i < countsView.extent(0); ++i) {
      constexpr std::size_t zero = 0;
      EXPECT_EQ(countsView_h(i), zero);
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool searchedValueExist) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(searchedValueExist, numTeams, numCols,
                                     apiId);
      }
    }
  }
}

TEST(std_algorithms_count_team_test, count_returns_nonzero) {
  constexpr bool searchedValueExist = true;
  run_all_scenarios<DynamicTag, double>(searchedValueExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValueExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValueExist);
}

TEST(std_algorithms_count_team_test, count_returns_zero) {
  constexpr bool searchedValueExist = false;
  run_all_scenarios<DynamicTag, double>(searchedValueExist);
  run_all_scenarios<StridedTwoRowsTag, int>(searchedValueExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(searchedValueExist);
}

}  // namespace TeamCount
}  // namespace stdalgos
}  // namespace Test
