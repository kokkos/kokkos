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
namespace TeamUniqueCopy {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct CustomPredicate {
  KOKKOS_INLINE_FUNCTION
  bool operator()(ValueType a, ValueType b) const { return a == b; }
};

template <class SourceViewType, class DestViewType, class DistancesViewType>
struct TestFunctorA {
  SourceViewType m_sourceView;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  int m_apiPick;

  TestFunctorA(const SourceViewType sourceView, const DestViewType destView,
               const DistancesViewType distancesView, int apiPick)
      : m_sourceView(sourceView),
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
          KE::unique_copy(member, KE::begin(myRowViewFrom),
                          KE::end(myRowViewFrom), KE::begin(myRowViewDest));

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::unique_copy(member, myRowViewFrom, myRowViewDest);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 2) {
      using comparator_t =
          CustomEqualityComparator<typename SourceViewType::value_type>;
      auto it = KE::unique_copy(member, KE::begin(myRowViewFrom),
                                KE::end(myRowViewFrom),
                                KE::begin(myRowViewDest), comparator_t());

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 3) {
      using comparator_t =
          CustomEqualityComparator<typename SourceViewType::value_type>;
      auto it =
          KE::unique_copy(member, myRowViewFrom, myRowViewDest, comparator_t());

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
     team-level KE::unique_copy on a rank-2 view where
     data is filled randomly such that we have several subsets
     of consecutive equal elements. Use one team per row.
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from a range that is tight enough that there is a high likelihood
  // of having several consecutive subsets of equal elements
  auto [sourceView, cloneOfDataViewBeforeOp_h] =
      create_random_view_and_host_clone(
          LayoutTag{}, numTeams, numCols,
          Kokkos::pair{ValueType(121), ValueType(153)}, "sourceView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // create the destination view
  Kokkos::View<ValueType**> destView("destView", numTeams, numCols);

  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expectation
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceView, destView, distancesView, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run std algo and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);
  Kokkos::View<ValueType**, Kokkos::HostSpace> stdDestView("stdDestView",
                                                           numTeams, numCols);

  for (std::size_t i = 0; i < cloneOfDataViewBeforeOp_h.extent(0); ++i) {
    auto myRowFrom =
        Kokkos::subview(cloneOfDataViewBeforeOp_h, i, Kokkos::ALL());
    auto myRowDest = Kokkos::subview(stdDestView, i, Kokkos::ALL());

    std::size_t stdDistance = 0;
    if (apiId <= 1) {
      auto it     = std::unique_copy(KE::cbegin(myRowFrom), KE::cend(myRowFrom),
                                 KE::begin(myRowDest));
      stdDistance = KE::distance(KE::begin(myRowDest), it);
    } else {
      auto it     = std::unique_copy(KE::cbegin(myRowFrom), KE::cend(myRowFrom),
                                 KE::begin(myRowDest),
                                 CustomEqualityComparator<value_type>{});
      stdDistance = KE::distance(KE::begin(myRowDest), it);
    }
    EXPECT_EQ(stdDistance, distancesView_h(i));
  }

  auto destViewAfterOp_h = create_host_space_copy(destView);
  expect_equal_host_views(stdDestView, destViewAfterOp_h);
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

TEST(std_algorithms_unique_copy_team_test, test) {
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, int>();
}

}  // namespace TeamUniqueCopy
}  // namespace stdalgos
}  // namespace Test
