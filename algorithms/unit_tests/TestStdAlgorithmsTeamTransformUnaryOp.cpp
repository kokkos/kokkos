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
namespace TeamTransformUnaryOp {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct PlusTwoUnaryOp {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& val) const {
    return val + static_cast<ValueType>(2);
  }
};

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

    using value_type = typename SourceViewType::value_type;
    if (m_apiPick == 0) {
      auto it = KE::transform(member, KE::cbegin(myRowViewFrom),
                              KE::cend(myRowViewFrom), KE::begin(myRowViewDest),
                              PlusTwoUnaryOp<value_type>());

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::transform(member, myRowViewFrom, myRowViewDest,
                              PlusTwoUnaryOp<value_type>());

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
     team level transform with each team handling a row of
     a rank-2 source view and applying a unary op that
     increments each element by two
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [sourceView, cloneOfSourceViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          Kokkos::pair{ValueType(0), ValueType(523)}, "sourceView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());
  // create the destination view
  Kokkos::View<ValueType**> destView("destView", numTeams, numCols);
  // make a host copy of the dest view that we can check below
  // to be all zeros since this should remain unchanged
  auto destViewBeforeOp_h = create_host_space_copy(destView);

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
  auto distancesView_h     = create_host_space_copy(distancesView);
  auto sourceViewAfterOp_h = create_host_space_copy(sourceView);
  auto destViewAfterOp_h   = create_host_space_copy(destView);
  for (std::size_t i = 0; i < destViewBeforeOp_h.extent(0); ++i) {
    for (std::size_t j = 0; j < destViewBeforeOp_h.extent(1); ++j) {
      // source view should not change
      EXPECT_EQ(sourceViewAfterOp_h(i, j), cloneOfSourceViewBeforeOp_h(i, j));

      // elements in dest view should be the source elements plus two
      EXPECT_EQ(destViewAfterOp_h(i, j), cloneOfSourceViewBeforeOp_h(i, j) + 2);
      EXPECT_EQ(destViewBeforeOp_h(i, j), ValueType(0));
    }

    // each team should return an iterator whose distance from the
    // beginning of the row equals the num of columns since
    // each team transforms all elements in each row
    EXPECT_TRUE(distancesView_h(i) == numCols);
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

TEST(std_algorithms_transform_team_test, test_unary_op) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

// TEST(std_algorithms_transform_team_test, test_binary_op) {
//   run_all_scenarios<DynamicTag, double>(1);
//   run_all_scenarios<StridedTwoRowsTag, int>(1);
//   run_all_scenarios<StridedThreeRowsTag, unsigned>(1);
// }

}  // namespace TeamTransformUnaryOp
}  // namespace stdalgos
}  // namespace Test
