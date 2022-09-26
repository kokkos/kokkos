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

namespace Test {
namespace stdalgos {
namespace TeamMismatch {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct EqualFunctor {
  KOKKOS_INLINE_FUNCTION bool operator()(const ValueType& lhs,
                                         const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DataViewType, class CompViewType, class ResultsViewType,
          class BinaryOpType>
struct TestFunctorA {
  DataViewType m_dataView;
  CompViewType m_compView;
  ResultsViewType m_resultsView;
  int m_apiPick;
  BinaryOpType m_binaryOp;

  TestFunctorA(const DataViewType dataView, const CompViewType compView,
               const ResultsViewType resultsView, int apiPick,
               BinaryOpType binaryOp)
      : m_dataView(dataView),
        m_compView(compView),
        m_resultsView(resultsView),
        m_apiPick(apiPick),
        m_binaryOp(binaryOp) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto rowIndex = member.league_rank();

    auto rowData   = Kokkos::subview(m_dataView, rowIndex, Kokkos::ALL());
    auto dataBegin = KE::begin(rowData);
    auto dataEnd   = KE::end(rowData);

    auto rowComp   = Kokkos::subview(m_compView, rowIndex, Kokkos::ALL());
    auto compBegin = KE::begin(rowComp);
    auto compEnd   = KE::end(rowComp);

    switch (m_apiPick) {
      case 0: {
        auto [dataIt, compIt] =
            KE::mismatch(member, dataBegin, dataEnd, compBegin, compEnd);

        const auto dataDist = KE::distance(dataBegin, dataIt);
        const auto compDist = KE::distance(compBegin, compIt);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_resultsView(rowIndex) = Kokkos::make_pair(dataDist, compDist);
        });

        break;
      }

      case 1: {
        const auto [dataIt, compIt] = KE::mismatch(
            member, dataBegin, dataEnd, compBegin, compEnd, m_binaryOp);

        const auto dataDist = KE::distance(dataBegin, dataIt);
        const auto compDist = KE::distance(compBegin, compIt);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_resultsView(rowIndex) = Kokkos::make_pair(dataDist, compDist);
        });

        break;
      }

      case 2: {
        const auto [dataIt, compIt] = KE::mismatch(member, rowData, rowComp);

        const std::size_t dataDist = KE::distance(dataBegin, dataIt);
        const std::size_t compDist = KE::distance(compBegin, compIt);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_resultsView(rowIndex) = Kokkos::make_pair(dataDist, compDist);
        });

        break;
      }

      case 3: {
        const auto [dataIt, compIt] =
            KE::mismatch(member, rowData, rowComp, m_binaryOp);

        const std::size_t dataDist = KE::distance(dataBegin, dataIt);
        const std::size_t compDist = KE::distance(compBegin, compIt);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_resultsView(rowIndex) = Kokkos::make_pair(dataDist, compDist);
        });

        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool viewsAreEqual, std::size_t numTeams, std::size_t numCols,
            int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level mismatch
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

  // create a view to compare it with dataView. If viewsAreEqual == true,
  // compView is a copy of dataView. If viewsAreEqual == false, compView is
  // randomly filled
  auto compView   = create_deep_copyable_compatible_clone(dataView);
  auto compView_h = create_mirror_view(Kokkos::HostSpace(), compView);
  if (viewsAreEqual) {
    Kokkos::deep_copy(compView_h, dataViewBeforeOp_h);
  } else {
    using rand_pool =
        Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    Kokkos::fill_random(compView_h, pool, lowerBound, upperBound);
  }

  Kokkos::deep_copy(compView, compView_h);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // create the view to store results of mismatch()
  Kokkos::View<Kokkos::pair<std::size_t, std::size_t>*> resultsView(
      "resultsView", numTeams);

  EqualFunctor<ValueType> binaryPred{};

  // use CTAD for functor
  TestFunctorA fnc(dataView, compView, resultsView, apiId, binaryPred);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto resultsView_h = create_host_space_copy(resultsView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowData = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());

    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    const std::size_t dataBeginEndDist = KE::distance(dataBegin, dataEnd);

    auto rowComp = Kokkos::subview(compView_h, i, Kokkos::ALL());

    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    const std::size_t compBeginEndDist = KE::distance(compBegin, compEnd);

    switch (apiId) {
      case 0:
      case 2: {
        const auto [dataIt, compIt] =
            std::mismatch(dataBegin, dataEnd, compBegin, compEnd);

        const std::size_t dataDist = KE::distance(dataBegin, dataIt);
        const std::size_t compDist = KE::distance(compBegin, compIt);

        if (viewsAreEqual) {
          EXPECT_EQ(dataBeginEndDist, resultsView_h(i).first);
          EXPECT_EQ(compBeginEndDist, resultsView_h(i).second);
        } else {
          EXPECT_EQ(dataDist, resultsView_h(i).first);
          EXPECT_EQ(compDist, resultsView_h(i).second);
        }

        break;
      }

      case 1:
      case 3: {
        const auto [dataIt, compIt] =
            std::mismatch(dataBegin, dataEnd, compBegin, compEnd, binaryPred);

        const std::size_t dataDist = KE::distance(dataBegin, dataIt);
        const std::size_t compDist = KE::distance(compBegin, compIt);

        if (viewsAreEqual) {
          EXPECT_EQ(dataBeginEndDist, resultsView_h(i).first);
          EXPECT_EQ(compBeginEndDist, resultsView_h(i).second);
        } else {
          EXPECT_EQ(dataDist, resultsView_h(i).first);
          EXPECT_EQ(compDist, resultsView_h(i).second);
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool viewsAreEqual) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(viewsAreEqual, numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_mismatch_team_test, views_are_equal) {
  constexpr bool viewsAreEqual = true;
  run_all_scenarios<DynamicTag, double>(viewsAreEqual);
  run_all_scenarios<StridedTwoRowsTag, int>(viewsAreEqual);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(viewsAreEqual);
}

TEST(std_algorithms_mismatch_team_test, views_are_not_equal) {
  constexpr bool viewsAreEqual = false;
  run_all_scenarios<DynamicTag, double>(viewsAreEqual);
  run_all_scenarios<StridedTwoRowsTag, int>(viewsAreEqual);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(viewsAreEqual);
}

}  // namespace TeamMismatch
}  // namespace stdalgos
}  // namespace Test
