//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <TestStdAlgorithmsCommon.hpp>

namespace Test {
namespace stdalgos {
namespace TeamGenerate_n {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct GenerateFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()() const { return static_cast<ValueType>(23); }
};

template <class ViewType, class DistancesViewType, class IntraTeamSentinelView>
struct TestFunctorA {
  ViewType m_view;
  DistancesViewType m_distancesView;
  IntraTeamSentinelView m_intraTeamSentinelView;
  std::size_t m_count;
  int m_apiPick;

  TestFunctorA(const ViewType view, const DistancesViewType distancesView,
               const IntraTeamSentinelView intraTeamSentinelView,
               std::size_t count, int apiPick)
      : m_view(view),
        m_distancesView(distancesView),
        m_intraTeamSentinelView(intraTeamSentinelView),
        m_count(count),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto leagueRank = member.league_rank();
    const auto myRowIndex = leagueRank;
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());
    ptrdiff_t resultDist  = 0;

    using value_type = typename ViewType::value_type;
    if (m_apiPick == 0) {
      auto it = KE::generate_n(member, KE::begin(myRowView), m_count,
                               GenerateFunctor<value_type>());

      resultDist = KE::distance(KE::begin(myRowView), it);
      Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) = resultDist;
      });
    } else if (m_apiPick == 1) {
      auto it    = KE::generate_n(member, myRowView, m_count,
                                  GenerateFunctor<value_type>());
      resultDist = KE::distance(KE::begin(myRowView), it);
      Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
        m_distancesView(myRowIndex) = resultDist;
      });
    }

    // store result of checking if all members have their local
    // values matching the one stored in m_distancesView
    member.team_barrier();
    const bool intraTeamCheck = team_members_have_matching_result(
        member, resultDist, m_distancesView(myRowIndex));
    Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
      m_intraTeamSentinelView(myRowIndex) = intraTeamCheck;
    });
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, std::size_t count,
            int apiId) {
  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range. Pick range so that it does NOT
  // contain the value produced by the generator (see top of file)
  // otherwise test check below is ill-posed
  auto [dataView, cloneOfDataViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          Kokkos::pair<ValueType, ValueType>{105, 523}, "dataView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expected value
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);
  // sentinel to check if all members of the team compute the same result
  Kokkos::View<bool*> intraTeamSentinelView("intraTeamSameResult", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, distancesView, intraTeamSentinelView, count,
                   apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto dataViewAfterOp_h       = create_host_space_copy(dataView);
  auto distancesView_h         = create_host_space_copy(distancesView);
  auto intraTeamSentinelView_h = create_host_space_copy(intraTeamSentinelView);
  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    // check that values match what we expect
    for (std::size_t j = 0; j < count; ++j) {
      ASSERT_EQ(dataViewAfterOp_h(i, j), static_cast<ValueType>(23));
      EXPECT_TRUE(dataViewAfterOp_h(i, j) != cloneOfDataViewBeforeOp_h(i, j));
    }
    // all other elements should be unchanged from before op
    for (std::size_t j = count; j < numCols; ++j) {
      ASSERT_EQ(dataViewAfterOp_h(i, j), cloneOfDataViewBeforeOp_h(i, j));
    }

    // check that returned iterators are correct
    if (count > 0) {
      ASSERT_EQ(distancesView_h(i), std::size_t(count));
    } else {
      ASSERT_EQ(distancesView_h(i), std::size_t(0));
    }
    ASSERT_TRUE(intraTeamSentinelView_h(i));
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  // prepare a map where, for a given set of num cols
  // we provide a list of counts of elements to generate
  // key = num of columns
  // value = list of num of elemenents to generate
  const std::map<int, std::vector<int>> scenarios = {
      {0, {0}},
      {2, {0, 1, 2}},
      {6, {0, 1, 2, 5}},
      {13, {0, 1, 2, 8, 11}},
      {56, {0, 1, 2, 8, 11, 33, 56}},
      {123, {0, 1, 11, 33, 56, 89, 112}}};

  for (int numTeams : teamSizesToTest) {
    for (const auto& scenario : scenarios) {
      const std::size_t numCols = scenario.first;
      for (int countToGenerate : scenario.second) {
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(numTeams, numCols, countToGenerate, apiId);
        }
      }
    }
  }
}

TEST(std_algorithms_generate_n_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamGenerate_n
}  // namespace stdalgos
}  // namespace Test
