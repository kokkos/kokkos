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
namespace TeamFindEnd {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct EqualFunctor {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DataViewType, class SearchedSequncesViewType,
          class DistancesViewType, class BinaryPredType>
struct TestFunctorA {
  DataViewType m_dataView;
  SearchedSequncesViewType m_searchedSequncesView;
  DistancesViewType m_distancesView;
  BinaryPredType m_binaryPred;
  int m_apiPick;

  TestFunctorA(const DataViewType dataView,
               const SearchedSequncesViewType searchedSequncesView,
               const DistancesViewType distancesView, BinaryPredType binaryPred,
               int apiPick)
      : m_dataView(dataView),
        m_searchedSequncesView(searchedSequncesView),
        m_distancesView(distancesView),
        m_binaryPred(binaryPred),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());
    auto myRowSearchedSeqView =
        Kokkos::subview(m_searchedSequncesView, myRowIndex, Kokkos::ALL());

    switch (m_apiPick) {
      case 0: {
        auto it = KE::find_end(
            member, KE::cbegin(myRowViewFrom), KE::cend(myRowViewFrom),
            KE::cbegin(myRowSearchedSeqView), KE::cend(myRowSearchedSeqView));
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });

        break;
      }

      case 1: {
        auto it = KE::find_end(member, myRowViewFrom, myRowSearchedSeqView);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::begin(myRowViewFrom), it);
        });

        break;
      }

      case 2: {
        auto it = KE::find_end(member, KE::cbegin(myRowViewFrom),
                               KE::cend(myRowViewFrom),
                               KE::cbegin(myRowSearchedSeqView),
                               KE::cend(myRowSearchedSeqView), m_binaryPred);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) =
              KE::distance(KE::cbegin(myRowViewFrom), it);
        });

        break;
      }

      case 3: {
        auto it = KE::find_end(member, myRowViewFrom, myRowSearchedSeqView,
                               m_binaryPred);
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
  const ValueType lowerBound{5}, upperBound{523};
  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, Kokkos::pair{lowerBound, upperBound},
      "dataView");

  // create a view that stores a sequence to found in dataView. If
  // sequencesExist == true it is filled base on dataView content, to allow
  // find_end to actually find anything. If sequencesExist == false it is filled
  // with random values greater than upperBound
  const auto halfCols    = (numCols + 1) / 2;
  const auto quarterCols = halfCols / 2;

  Kokkos::View<ValueType**> searchedSequncesView(
      "searchedSequncesView", numTeams, halfCols - quarterCols);
  auto searchedSequncesView_h = create_host_space_copy(searchedSequncesView);

  if (sequencesExist) {
    for (std::size_t i = 0; i < searchedSequncesView_h.extent(0); ++i) {
      for (std::size_t js = 0, jd = quarterCols; jd < halfCols; ++js, ++jd) {
        searchedSequncesView_h(i, js) = dataViewBeforeOp_h(i, jd);
      }
    }
  } else {
    using rand_pool =
        Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    Kokkos::fill_random(searchedSequncesView_h, pool, upperBound,
                        upperBound * 2);
  }

  Kokkos::deep_copy(searchedSequncesView, searchedSequncesView_h);

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

  EqualFunctor<ValueType> binaryPred;

  // use CTAD for functor
  TestFunctorA fnc(dataView, searchedSequncesView, distancesView, binaryPred,
                   apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());
    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);

    auto rowSearchedSeq =
        Kokkos::subview(searchedSequncesView_h, i, Kokkos::ALL());

    std::size_t stdDistance;
    const std::size_t beginEndDistance = KE::distance(rowFromBegin, rowFromEnd);

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
    }

    if (sequencesExist) {
      EXPECT_LT(stdDistance, beginEndDistance);
    } else {
      EXPECT_EQ(stdDistance, beginEndDistance);
    }

    EXPECT_EQ(stdDistance, distancesView_h(i));
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
