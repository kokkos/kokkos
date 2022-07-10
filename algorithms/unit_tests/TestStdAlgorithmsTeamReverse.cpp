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
#include <Kokkos_Random.hpp>

namespace Test {
namespace stdalgos {
namespace TeamReverse {

namespace KE = Kokkos::Experimental;

template <class ViewType, class MemberType>
struct FunctorA {
  ViewType m_view;
  int m_api_pick;

  FunctorA(const ViewType view, int apiPick)
      : m_view(view), m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    if (m_api_pick == 0) {
      KE::reverse(member, KE::begin(myRowView), KE::end(myRowView));
    } else if (m_api_pick == 1) {
      KE::reverse(member, myRowView);
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols, int apiId) {
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

  auto gold = create_host_space_copy(v);
  for (std::size_t i = 0; i < gold.extent(0); ++i) {
    for (std::size_t j = 0; j < gold.extent(1) / 2; ++j) {
      std::swap(gold(i, j), gold(i, num_cols - j - 1));
    }
  }

  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  using functor_type = FunctorA<decltype(v), team_member_type>;
  functor_type fnc(v, apiId);
  Kokkos::parallel_for(policy, fnc);

  auto v_h = create_host_space_copy(v);
  for (std::size_t i = 0; i < v_h.extent(0); ++i) {
    for (std::size_t j = 0; j < v_h.extent(1); ++j) {
      EXPECT_TRUE(v_h(i, j) == gold(i, j));
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (int num_teams : team_sizes_to_test) {
    for (const auto& scenario : default_scenarios) {
      for (int apiId : {0, 1}) {
        test_A<Tag, ValueType>(num_teams, scenario.second, apiId);
      }
    }
  }
}

TEST(std_algorithms_reverse_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReverse
}  // namespace stdalgos
}  // namespace Test
