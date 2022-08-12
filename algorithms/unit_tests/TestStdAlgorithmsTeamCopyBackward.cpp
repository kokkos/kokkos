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
namespace TeamCopybackward {

namespace KE = Kokkos::Experimental;

template <class SourceViewType, class DestViewType, class DistancesViewType>
struct TestFunctorA {
  SourceViewType m_sourceView;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const SourceViewType fromView, const DestViewType destView,
               const DistancesViewType distancesView, int apiPick)
      : m_sourceView(fromView),
        m_destView(destView),
        m_distancesView(distancesView),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom =
        Kokkos::subview(m_sourceView, myRowIndex, Kokkos::ALL());
    auto myRowViewDest = Kokkos::subview(m_destView, myRowIndex, Kokkos::ALL());

    if (m_apiPick == 0) {
      auto it =
          KE::copy_backward(member, KE::cbegin(myRowViewFrom),
                            KE::cend(myRowViewFrom), KE::end(myRowViewDest));

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::copy_backward(member, myRowViewFrom, myRowViewDest);
      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     randomly fill a source view and copy_backward into a destination view.
     The operation is done via a team parfor with one row per team.
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [sourceView, sourceViewBeforeOp_h] = create_view_and_fill_randomly(
      LayoutTag{}, numTeams, numCols,
      Kokkos::pair{ValueType(11), ValueType(523)}, "sourceView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());
  // create the destination view: for a meaningful test, the destination
  // view must have more columns that than the source view so that we
  // can check that the elements are copied into the right place
  constexpr std::size_t extra = 10;
  Kokkos::View<ValueType**> destView("destView", numTeams, numCols + extra);
  // make a host copy of the destination view that should be unchanged after the
  // op
  auto destViewBeforeOp_h = create_host_space_copy(destView);

  // copy_backward returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceView, destView, distancesView, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto distancesView_h   = create_host_space_copy(distancesView);
  auto destViewAfterOp_h = create_host_space_copy(destView);
  for (std::size_t i = 0; i < destViewAfterOp_h.extent(0); ++i) {
    // first extra num of columns should be unchanged
    for (std::size_t j = 0; j < extra; ++j) {
      EXPECT_TRUE(destViewAfterOp_h(i, j) == destViewBeforeOp_h(i, j));
    }

    // after extra # of column (inclusive) should match the source view
    for (std::size_t j = extra; j < destViewBeforeOp_h.extent(1); ++j) {
      EXPECT_TRUE(sourceViewBeforeOp_h(i, j - extra) ==
                  destViewAfterOp_h(i, j));
    }

    // each team should have returned an interator whose distance
    // from the beginning of the row should satisfy this
    EXPECT_TRUE(distancesView_h(i) == extra);
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 11113}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_copy_backward_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamCopybackward
}  // namespace stdalgos
}  // namespace Test
