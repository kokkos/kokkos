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
namespace TeamGenerate {

namespace KE = Kokkos::Experimental;

template<class ValueType>
struct GenerateFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()() const {
    return static_cast<ValueType>(23);
  }
};

template <class ViewFromType, class MemberType>
struct TestFunctorA {
  ViewFromType m_from_view;
  int m_api_pick;

  TestFunctorA(const ViewFromType viewFrom, int apiPick)
      : m_from_view(viewFrom), m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView = Kokkos::subview(m_from_view, myRowIndex, Kokkos::ALL());

    using value_type = typename ViewFromType::value_type;
    if (m_api_pick == 0) {
      KE::generate(member, KE::begin(myRowView), KE::end(myRowView),
		   GenerateFunctor<value_type>());
    } else if (m_api_pick == 1) {
      KE::generate(member, myRowView, GenerateFunctor<value_type>());
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols, int apiId) {
  /* description: */

  // v constructed on memory space associated with default exespace
  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");

  // launch kernel
  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  using functor_type = TestFunctorA<decltype(v), team_member_type>;
  functor_type fnc(v, apiId);
  Kokkos::parallel_for(policy, fnc);

  // check
  auto v_h  = create_host_space_copy(v);
  for (std::size_t i = 0; i < v_h.extent(0); ++i) {
    for (std::size_t j = 0; j < v_h.extent(1); ++j) {
      EXPECT_TRUE(v_h(i, j) == static_cast<ValueType>(23));
    }
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

TEST(std_algorithms_generate_team_test, test_unary_op) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamGenerate
}  // namespace stdalgos
}  // namespace Test
