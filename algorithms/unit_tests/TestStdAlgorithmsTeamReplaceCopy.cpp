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
namespace TeamReplaceCopy {

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

template <class SourceViewType, class DestViewType, class DistancesViewType,
          class ValueType>
struct TestFunctorA {
  SourceViewType m_sourceView;
  DestViewType m_destView;
  DistancesViewType m_distancesView;
  ValueType m_targetValue;
  ValueType m_newValue;
  int m_apiPick;

  TestFunctorA(const SourceViewType fromView, const DestViewType destView,
               const DistancesViewType distancesView, ValueType targetVal,
               ValueType newVal, int apiPick)
      : m_sourceView(fromView),
        m_destView(destView),
        m_distancesView(distancesView),
        m_targetValue(targetVal),
        m_newValue(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom =
        Kokkos::subview(m_sourceView, myRowIndex, Kokkos::ALL());
    auto myRowViewDest = Kokkos::subview(m_destView, myRowIndex, Kokkos::ALL());

    if (m_apiPick == 0) {
      auto it = KE::replace_copy(
          member, KE::begin(myRowViewFrom), KE::end(myRowViewFrom),
          KE::begin(myRowViewDest), m_targetValue, m_newValue);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::replace_copy(member, myRowViewFrom, myRowViewDest,
                                 m_targetValue, m_newValue);
      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) =
            KE::distance(KE::begin(myRowViewDest), it);
      });
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a "source" and "destination" rank-2 views such that in the source,
     for each row, a random subset of elements is filled with a target value
     that we want to replace_copy with a new value into the destination view.
     The operation is done via a team policy with one row per team
     to test the team level KE::replace_copy().
     Basically the same as KE::replace expect that we don't modify the source
     view.
   */

  const auto targetVal = static_cast<ValueType>(531);
  const auto newVal    = static_cast<ValueType>(123);

  //
  // prepare data
  //
  // construct in memory space associated with default exespace
  auto sourceView =
      create_view<ValueType>(Tag{}, numTeams, numCols, "sourceView");

  // sourceView might not deep copyable (e.g. strided layout) so to fill it
  // we make a new view that is for sure deep copyable, modify it on the host
  // deep copy to device and then launch copy kernel to sourceView
  auto sourceView_dc =
      create_deep_copyable_compatible_view_with_same_extent(sourceView);
  auto sourceView_dc_h = create_mirror_view(Kokkos::HostSpace(), sourceView_dc);

  // for each row, randomly select columns, fill with targetVal
  const std::size_t maxColInd = numCols > 0 ? numCols - 1 : 0;
  UnifDist<int> colCountProducer(maxColInd, 3123377);
  UnifDist<int> colIndicesProducer(maxColInd, 455225);
  for (std::size_t i = 0; i < sourceView_dc_h.extent(0); ++i) {
    const std::size_t currCount = colCountProducer();
    for (std::size_t j = 0; j < currCount; ++j) {
      const auto colInd          = colIndicesProducer();
      sourceView_dc_h(i, colInd) = targetVal;
    }
  }

  // copy to sourceView_dc and then to sourceView
  Kokkos::deep_copy(sourceView_dc, sourceView_dc_h);
  // use CTAD
  CopyFunctorRank2 F1(sourceView_dc, sourceView);
  Kokkos::parallel_for("copy", sourceView.extent(0) * sourceView.extent(1), F1);

  //
  // launch kokkos kernel
  //
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());
  // create the destination view where we to store the replace_copy
  Kokkos::View<ValueType**> destView("destView", numTeams, numCols);

  // replace_copy returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the std result
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(sourceView, destView, distancesView, targetVal, newVal,
                   apiId);
  Kokkos::parallel_for(policy, fnc);

  //
  // run cpp-std kernel and check
  //
  Kokkos::View<ValueType**> stdDestView("stdDestView", numTeams, numCols);
  for (std::size_t i = 0; i < sourceView_dc_h.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(sourceView_dc_h, i, Kokkos::ALL());
    auto rowDest = Kokkos::subview(stdDestView, i, Kokkos::ALL());
    auto it      = std::replace_copy(KE::cbegin(rowFrom), KE::cend(rowFrom),
                                KE::begin(rowDest), targetVal, newVal);
    const std::size_t stdDistance = KE::distance(KE::begin(rowDest), it);
    EXPECT_EQ(stdDistance, distancesView(i));
  }

  auto dataViewAfterOp_h = create_host_space_copy(destView);
  expect_equal_host_views(stdDestView, dataViewAfterOp_h);
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : team_sizes_to_test) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 11113}) {
      for (int apiId : {0, 1}) {
        test_A<Tag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_replace_copy_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplaceCopy
}  // namespace stdalgos
}  // namespace Test
