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
namespace TeamReplace {

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

template <class ViewType, class ValueType>
struct TestFunctorA {
  ViewType m_view;
  ValueType m_targetValue;
  ValueType m_newValue;
  int m_apiPick;

  TestFunctorA(const ViewType view, ValueType oldVal, ValueType newVal,
               int apiPick)
      : m_view(view),
        m_targetValue(oldVal),
        m_newValue(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    if (m_apiPick == 0) {
      KE::replace(member, KE::begin(myRowView), KE::end(myRowView),
                  m_targetValue, m_newValue);
    } else if (m_apiPick == 1) {
      KE::replace(member, myRowView, m_targetValue, m_newValue);
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     set a random subset of each row of a rank-2 view
     to a target value that we want to replace with a new value.
     Do the operation via a team parfor with one row per team.
   */

  const auto targetVal = static_cast<ValueType>(531);
  const auto newVal    = static_cast<ValueType>(123);

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // construct in memory space associated with default exespace
  auto dataView =
      create_view<ValueType>(LayoutTag{}, numTeams, numCols, "dataView");

  // dataView might not deep copyable (e.g. strided layout) so to fill it
  // we make a new view that is for sure deep copyable, modify it on the host
  // deep copy to device and then launch copy kernel to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  // for each row, randomly select columns, fill with targetVal
  // and for testing purposes keep track of the entries we are changing.
  // To do this, I need one rand num obj to generate how many elements
  // and one object to generate the actual indices to pick.

  std::vector<std::size_t> targetElementsLinearizedIds;
  const std::size_t maxColInd = numCols > 0 ? numCols - 1 : 0;
  UnifDist<int> colCountProducer(maxColInd, 3123377);
  UnifDist<int> colIndicesProducer(maxColInd, 455225);
  for (std::size_t i = 0; i < dataView_dc_h.extent(0); ++i) {
    const std::size_t currCount = colCountProducer();
    for (std::size_t j = 0; j < currCount; ++j) {
      const auto colInd        = colIndicesProducer();
      dataView_dc_h(i, colInd) = targetVal;
      targetElementsLinearizedIds.push_back(i * numCols + colInd);
    }
  }

  // copy to dataView_dc and then to dataView
  Kokkos::deep_copy(dataView_dc, dataView_dc_h);
  CopyFunctorRank2 F1(dataView_dc, dataView);
  Kokkos::parallel_for("copy", dataView.extent(0) * dataView.extent(1), F1);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());
  // use CTAD for functor
  TestFunctorA fnc(dataView, targetVal, newVal, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // conditions for test passing:
  // - the target elements are replaced with the new value
  // - all other elements are unchanged
  // -----------------------------------------------

  // make a copy on host (generic to handle a non deep-copyable view)
  // check that the correct elements have the new value
  auto dataView2_h = create_host_space_copy(dataView);
  for (auto k : targetElementsLinearizedIds) {
    const std::size_t i = k / numCols;
    const std::size_t j = k % numCols;
    EXPECT_TRUE(dataView2_h(i, j) == newVal);
  }

  // figure out which elements should be unchanged
  std::vector<std::size_t> allIndices(numTeams * numCols);
  std::iota(allIndices.begin(), allIndices.end(), 0);
  std::vector<std::size_t> unchanged(allIndices.size());
  // set_difference requires sorted ranges, allIndices is already sorted
  std::sort(targetElementsLinearizedIds.begin(),
            targetElementsLinearizedIds.end());
  auto bound = std::set_difference(allIndices.cbegin(), allIndices.cend(),
                                   targetElementsLinearizedIds.cbegin(),
                                   targetElementsLinearizedIds.cend(),
                                   unchanged.begin());

  auto verify = [numCols, dataView2_h](std::size_t k) {
    const std::size_t i = k / numCols;
    const std::size_t j = k % numCols;
    EXPECT_TRUE(dataView2_h(i, j) == static_cast<ValueType>(0));
  };
  std::for_each(unchanged.begin(), bound, verify);
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

TEST(std_algorithms_replace_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplace
}  // namespace stdalgos
}  // namespace Test
