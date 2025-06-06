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
namespace TeamFindEnd {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct EqualFunctor {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DataViewType, class SearchedSequencesViewType,
          class DistancesViewType, class IntraTeamSentinelView,
          class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  SearchedSequencesViewType m_searchedSequencesView;
  DistancesViewType m_distancesView;
  IntraTeamSentinelView m_intraTeamSentinelView;
  BinaryPredType m_binaryPred;
  int m_apiPick;

  TestFunctorA(const DataViewType dataView,
               const SearchedSequencesViewType searchedSequencesView,
               const DistancesViewType distancesView,
               const IntraTeamSentinelView intraTeamSentinelView,
               BinaryPredType binaryPred, int apiPick)
      : m_dataView(dataView),
        m_searchedSequencesView(searchedSequencesView),
        m_distancesView(distancesView),
        m_intraTeamSentinelView(intraTeamSentinelView),
        m_binaryPred(binaryPred),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());
    auto myRowSearchedSeqView =
        Kokkos::subview(m_searchedSequencesView, myRowIndex, Kokkos::ALL());
    ptrdiff_t resultDist = 0;

    switch (m_apiPick) {
      case 0: {
        auto it = KE::find_end(
            member, KE::cbegin(myRowViewFrom), KE::cend(myRowViewFrom),
            KE::cbegin(myRowSearchedSeqView), KE::cend(myRowSearchedSeqView));
        resultDist = KE::distance(KE::cbegin(myRowViewFrom), it);
        Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = resultDist;
        });

        break;
      }

      case 1: {
        auto it    = KE::find_end(member, myRowViewFrom, myRowSearchedSeqView);
        resultDist = KE::distance(KE::begin(myRowViewFrom), it);
        Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = resultDist;
        });

        break;
      }

      case 2: {
        auto it    = KE::find_end(member, KE::cbegin(myRowViewFrom),
                                  KE::cend(myRowViewFrom),
                                  KE::cbegin(myRowSearchedSeqView),
                                  KE::cend(myRowSearchedSeqView), m_binaryPred);
        resultDist = KE::distance(KE::cbegin(myRowViewFrom), it);
        Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = resultDist;
        });

        break;
      }

      case 3: {
        auto it    = KE::find_end(member, myRowViewFrom, myRowSearchedSeqView,
                                  m_binaryPred);
        resultDist = KE::distance(KE::begin(myRowViewFrom), it);
        Kokkos::single(Kokkos::PerTeam(member), [=, *this]() {
          m_distancesView(myRowIndex) = resultDist;
        });

        break;
      }
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
void test_A(const bool sequencesExist, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level find_end
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  constexpr ValueType lowerBound = 5;
  constexpr ValueType upperBound = 523;
  const auto bounds              = make_bounds(lowerBound, upperBound);

  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, bounds, "dataView");

  // create a view that stores a sequence to found in dataView. If
  // sequencesExist == true it is filled base on dataView content, to allow
  // find_end to actually find anything. If sequencesExist == false it is filled
  // with random values greater than upperBound
  const auto halfCols    = (numCols + 1) / 2;
  const auto quarterCols = halfCols / 2;

  Kokkos::View<ValueType**> searchedSequencesView(
      "searchedSequencesView", numTeams, halfCols - quarterCols);
  auto searchedSequencesView_h = create_host_space_copy(searchedSequencesView);

  if (sequencesExist) {
    for (std::size_t i = 0; i < searchedSequencesView_h.extent(0); ++i) {
      for (std::size_t js = 0, jd = quarterCols; jd < halfCols; ++js, ++jd) {
        searchedSequencesView_h(i, js) = dataViewBeforeOp_h(i, jd);
      }
    }
  } else {
    using rand_pool =
        Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    Kokkos::fill_random(searchedSequencesView_h, pool, upperBound,
                        upperBound * 2);
  }

  Kokkos::deep_copy(searchedSequencesView, searchedSequencesView_h);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // find_end returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);
  // sentinel to check if all members of the team compute the same result
  Kokkos::View<bool*> intraTeamSentinelView("intraTeamSameResult", numTeams);

  EqualFunctor<ValueType> binaryPred;

  // use CTAD for functor
  TestFunctorA fnc(dataView, searchedSequencesView, distancesView,
                   intraTeamSentinelView, binaryPred, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h         = create_host_space_copy(distancesView);
  auto intraTeamSentinelView_h = create_host_space_copy(intraTeamSentinelView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());
    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);

    auto rowSearchedSeq =
        Kokkos::subview(searchedSequencesView_h, i, Kokkos::ALL());

    std::size_t stdDistance = std::numeric_limits<std::size_t>::max();
    const std::size_t beginEndDistance = KE::distance(rowFromBegin, rowFromEnd);

    ASSERT_TRUE(intraTeamSentinelView_h(i));
    switch (apiId) {
      case 0:
      case 1: {
        auto it =
            std::find_end(rowFromBegin, rowFromEnd, KE::cbegin(rowSearchedSeq),
                          KE::cend(rowSearchedSeq));
        stdDistance = KE::distance(rowFromBegin, it);

        break;
      }

      case 2:
      case 3: {
        auto it =
            std::find_end(rowFromBegin, rowFromEnd, KE::cbegin(rowSearchedSeq),
                          KE::cend(rowSearchedSeq), binaryPred);
        stdDistance = KE::distance(rowFromBegin, it);

        break;
      }
      default: Kokkos::abort("unreachable");
    }

    if (sequencesExist) {
      EXPECT_LT(stdDistance, beginEndDistance);
    } else {
      ASSERT_EQ(stdDistance, beginEndDistance);
    }

    ASSERT_EQ(stdDistance, distancesView_h(i));
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool sequencesExist) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(sequencesExist, numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_find_end_team_test, sequences_exist) {
  constexpr bool sequencesExist = true;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

TEST(std_algorithms_find_end_team_test, sequences_do_not_exist) {
  constexpr bool sequencesExist = false;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

}  // namespace TeamFindEnd
}  // namespace stdalgos
}  // namespace Test
