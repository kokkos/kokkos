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
#include <Kokkos_Random.hpp>

namespace Test {
namespace stdalgos {
namespace TeamCopy_n {

namespace KE = Kokkos::Experimental;

template <class ViewFromType, class ViewDestType, class MemberType>
struct TestFunctorA {
  ViewFromType m_from_view;
  ViewDestType m_dest_view;
  int m_api_pick;
  int m_num_to_copy;

  TestFunctorA(const ViewFromType viewFrom, const ViewDestType viewDest,
               int apiPick, int n_to_copy)
    : m_from_view(viewFrom), m_dest_view(viewDest), m_api_pick(apiPick),
      m_num_to_copy(n_to_copy){}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom =
        Kokkos::subview(m_from_view, myRowIndex, Kokkos::ALL());
    auto myRowViewDest =
        Kokkos::subview(m_dest_view, myRowIndex, Kokkos::ALL());

    if (m_api_pick == 0) {
      auto it = KE::copy_n(member, KE::begin(myRowViewFrom), m_num_to_copy, KE::begin(myRowViewDest));
      (void)it;
    } else if (m_api_pick == 1) {
      auto it = KE::copy_n(member, myRowViewFrom, m_num_to_copy, myRowViewDest);
      (void)it;
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols, std::size_t n_to_copy, int apiId)
{
  /* description:
     fill randomly a matrix and copy to another matrix
     using a team par_for where each team handles one row
   */

  // v constructed on memory space associated with default exespace
  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");

  // v might not deep copyable so to modify it on the host
  auto v_dc   = create_deep_copyable_compatible_view_with_same_extent(v);
  auto v_dc_h = create_mirror_view(Kokkos::HostSpace(), v_dc);

  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(12371);
  Kokkos::fill_random(v_dc_h, pool, 0, 523);
  // copy to v_dc and then to v
  Kokkos::deep_copy(v_dc, v_dc_h);
  CopyFunctorRank2<decltype(v_dc), decltype(v)> F1(v_dc, v);
  Kokkos::parallel_for("copy", v.extent(0) * v.extent(1), F1);

  // launch kernel
  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  auto v2 = create_view<ValueType>(Tag{}, num_teams, num_cols, "v2");
  using functor_type =
      TestFunctorA<decltype(v), decltype(v2), team_member_type>;
  functor_type fnc(v, v2, apiId, n_to_copy);
  Kokkos::parallel_for(policy, fnc);

  // check
  auto v_h  = create_host_space_copy(v);
  auto v2_h = create_host_space_copy(v2);
  for (std::size_t i = 0; i < v_h.extent(0); ++i) {
    for (std::size_t j = 0; j < v_h.extent(1); ++j) {
      if (j < n_to_copy) {
	EXPECT_TRUE(v_h(i, j) == v2_h(i, j));
      } else {
        EXPECT_TRUE(v2_h(i, j) == static_cast<ValueType>(0));
      }
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios()
{
  // key = num of columns,
  // value = list of num of elemenents to fill
  using v_t                          = std::vector<int>;
  const std::map<int, v_t> scenarios = {{0, v_t{0}},
                                        {2, v_t{0, 1, 2}},
                                        {6, v_t{0, 1, 2, 5}},
                                        {13, v_t{0, 1, 2, 8, 11}}};

  for (int num_teams : team_sizes_to_test) {
    for (const auto& scenario : scenarios) {
      const std::size_t numCols = scenario.first;
      for (int numElementsToCopy : scenario.second) {
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(num_teams, numCols, numElementsToCopy, apiId);
        }
      }
    }
  }
}

TEST(std_algorithms_copy_n_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamCopy_n
}  // namespace stdalgos
}  // namespace Test
