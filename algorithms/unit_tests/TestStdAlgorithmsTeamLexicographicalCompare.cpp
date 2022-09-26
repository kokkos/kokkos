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
namespace TeamLexicographicalCompare {

namespace KE = Kokkos::Experimental;

enum class TestCaseType { ViewsAreEqual, FirstIsLess, FirstIsGreater };

template <class ValueType>
struct LessFunctor {
  KOKKOS_INLINE_FUNCTION bool operator()(const ValueType& lhs,
                                         const ValueType& rhs) const {
    return lhs < rhs;
  }
};

template <class DataViewType, class CompViewType, class ResultsViewType,
          class BinaryCompType>
struct TestFunctorA {
  DataViewType m_dataView;
  CompViewType m_compView;
  ResultsViewType m_resultsView;
  int m_apiPick;
  BinaryCompType m_binaryComp;

  TestFunctorA(const DataViewType dataView, const CompViewType compView,
               const ResultsViewType resultsView, int apiPick,
               BinaryCompType binaryComp)
      : m_dataView(dataView),
        m_compView(compView),
        m_resultsView(resultsView),
        m_apiPick(apiPick),
        m_binaryComp(binaryComp) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto rowIndex = member.league_rank();

    auto rowData         = Kokkos::subview(m_dataView, rowIndex, Kokkos::ALL());
    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    auto rowComp         = Kokkos::subview(m_compView, rowIndex, Kokkos::ALL());
    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    switch (m_apiPick) {
      case 0: {
        const bool result = KE::lexicographical_compare(
            member, dataBegin, dataEnd, compBegin, compEnd);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 1: {
        const bool result =
            KE::lexicographical_compare(member, rowData, rowComp);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 2: {
        const bool result = KE::lexicographical_compare(
            member, dataBegin, dataEnd, compBegin, compEnd, m_binaryComp);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }

      case 3: {
        const bool result =
            KE::lexicographical_compare(member, rowData, rowComp, m_binaryComp);
        Kokkos::single(Kokkos::PerTeam(member),
                       [=]() { m_resultsView(rowIndex) = result; });
        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const TestCaseType testCase, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level lexicographical_compare
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

  // create a view to compare it with dataView. If testCase == ViewsAreEqual,
  // compView is a copy of dataView. If testCase == FirstIsLess, we want the
  // dataView to be lexicographically less (and compView - greater). If testCase
  // == FirstIsGreater, we want the dataView to be lexicographically greater
  // (and compView - less).
  auto compEqualView   = create_deep_copyable_compatible_clone(dataView);
  auto compEqualView_h = create_mirror_view(Kokkos::HostSpace(), compEqualView);
  Kokkos::deep_copy(compEqualView_h, dataViewBeforeOp_h);
  const auto middle = numCols / 2;
  switch (testCase) {
    case TestCaseType::ViewsAreEqual: {
      // Do nothing - deep_copy was already done
      break;
    }

    case TestCaseType::FirstIsLess: {
      for (std::size_t i = 0; i < compEqualView_h.extent(0); ++i) {
        compEqualView_h(i, middle) += 1;
      }

      break;
    }

    case TestCaseType::FirstIsGreater: {
      for (std::size_t i = 0; i < compEqualView_h.extent(0); ++i) {
        compEqualView_h(i, middle) -= 1;
      }

      break;
    }
  }

  Kokkos::deep_copy(compEqualView, compEqualView_h);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // create the view to store results of equal()
  Kokkos::View<bool*> resultsView("resultsView", numTeams);

  LessFunctor<ValueType> binaryComp{};

  // use CTAD for functor
  TestFunctorA fnc(dataView, compEqualView, resultsView, apiId, binaryComp);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto resultsView_h = create_host_space_copy(resultsView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowData = Kokkos::subview(dataViewBeforeOp_h, i, Kokkos::ALL());
    const auto dataBegin = KE::cbegin(rowData);
    const auto dataEnd   = KE::cend(rowData);

    auto rowComp         = Kokkos::subview(compEqualView_h, i, Kokkos::ALL());
    const auto compBegin = KE::cbegin(rowComp);
    const auto compEnd   = KE::cend(rowComp);

    switch (apiId) {
      case 0:
      case 1: {
        const bool result = std::lexicographical_compare(dataBegin, dataEnd,
                                                         compBegin, compEnd);

        switch (testCase) {
          case TestCaseType::ViewsAreEqual:
          case TestCaseType::FirstIsGreater: {
            EXPECT_FALSE(resultsView_h(i));
            EXPECT_EQ(result, resultsView_h(i));
            break;
          }

          case TestCaseType::FirstIsLess: {
            EXPECT_TRUE(resultsView_h(i));
            EXPECT_EQ(result, resultsView_h(i));
            break;
          }
        }

        break;
      }

      case 2:
      case 3: {
        const bool result = std::lexicographical_compare(
            dataBegin, dataEnd, compBegin, compEnd, binaryComp);

        switch (testCase) {
          case TestCaseType::ViewsAreEqual:
          case TestCaseType::FirstIsGreater: {
            EXPECT_FALSE(resultsView_h(i));
            EXPECT_EQ(result, resultsView_h(i));
            break;
          }

          case TestCaseType::FirstIsLess: {
            EXPECT_TRUE(resultsView_h(i));
            EXPECT_EQ(result, resultsView_h(i));
            break;
          }
        }

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const TestCaseType testCase) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1, 2, 3}) {
        test_A<LayoutTag, ValueType>(testCase, numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_lexicographical_compare_team_test, views_are_equal) {
  constexpr TestCaseType testCaseType = TestCaseType::ViewsAreEqual;
  run_all_scenarios<DynamicTag, double>(testCaseType);
  run_all_scenarios<StridedTwoRowsTag, int>(testCaseType);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(testCaseType);
}

TEST(std_algorithms_lexicographical_compare_team_test, first_view_is_less) {
  constexpr TestCaseType testCaseType = TestCaseType::FirstIsLess;
  run_all_scenarios<DynamicTag, double>(testCaseType);
  run_all_scenarios<StridedTwoRowsTag, int>(testCaseType);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(testCaseType);
}

TEST(std_algorithms_lexicographical_compare_team_test, first_view_is_greater) {
  constexpr TestCaseType testCaseType = TestCaseType::FirstIsGreater;
  run_all_scenarios<DynamicTag, double>(testCaseType);
  run_all_scenarios<StridedTwoRowsTag, int>(testCaseType);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(testCaseType);
}

}  // namespace TeamLexicographicalCompare
}  // namespace stdalgos
}  // namespace Test
