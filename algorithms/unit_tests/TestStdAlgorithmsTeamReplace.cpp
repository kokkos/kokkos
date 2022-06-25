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

template <class ViewType, class MemberType, class ValueType>
struct TestFunctorA {
  ViewType m_view;
  ValueType m_targetValue;
  int m_api_pick;

  TestFunctorA(const ViewType view, ValueType val, int apiPick)
    : m_view(view), m_targetValue(val), m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());
    const auto newValue = static_cast<ValueType>(123);

    if (m_api_pick == 0) {
      KE::replace(member, KE::begin(myRowView), KE::end(myRowView), m_targetValue, newValue);
    } else if (m_api_pick == 1) {
      KE::replace(member, myRowView, m_targetValue, newValue);
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols, int apiId) {
  /* description:
     start from a matrix where a random subset of elements in
     each row is filled with a "target" value that we want to replace.
     The replace is done via a team policy with one row per team,
     the team calls replace on that row
     such that the "target" value is replaced with a new one.
   */

  const auto targetVal = static_cast<ValueType>(531);

  // v constructed on memory space associated with default exespace
  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");

  // v might not deep copyable so to modify it on the host
  auto v_h = create_host_space_copy(v);

  std::vector<std::size_t> rowIndOfTargetElements;
  std::vector<std::size_t> colIndOfTargetElements;
  // I need one rand num obj to generate how many cols to change
  // and one object to produce the random indices to change
  const std::size_t maxColInd = v_h.extent(1) > 0 ? v_h.extent(1)-1 : 0;
  UnifDist<int> howManyColsToChangeProducer(maxColInd, 3123377);
  UnifDist<int> colIndicesProducer(maxColInd, 455225);
  for (std::size_t i=0; i<v_h.extent(0); ++i)
  {
    const std::size_t numToChange = howManyColsToChangeProducer();
    std::vector<std::size_t> colIndices(numToChange);
    for (std::size_t k=0; k<numToChange; ++k){
      colIndices[k] = colIndicesProducer();
    }

    for (std::size_t j : colIndices){
      v_h(i, j) = targetVal;
      rowIndOfTargetElements.push_back(i);
      colIndOfTargetElements.push_back(j);
    }
  }

  // copy from v_h to v (deep copy might not be applicable)
  CopyFunctorRank2<decltype(v_h), decltype(v)> F1(v_h, v);
  Kokkos::parallel_for("copy", v.extent(0)*v.extent(1), F1);

  // launch kernel
  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  using functor_type = TestFunctorA<decltype(v), team_member_type, ValueType>;
  functor_type fnc(v, targetVal, apiId);
  Kokkos::parallel_for(policy, fnc);

  // check
  auto v2_h = create_host_space_copy(v);
  for (std::size_t k=0; k<rowIndOfTargetElements.size(); ++k){
    EXPECT_TRUE(v2_h(rowIndOfTargetElements[k], colIndOfTargetElements[k])
		== static_cast<ValueType>(123));
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (int num_teams : team_sizes_to_test) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 51153}) {
      for (int apiId : {0, 1}) {
	test_A<Tag, ValueType>(num_teams, numCols, apiId);
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
