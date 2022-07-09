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
namespace TeamGenerate_n {

namespace KE = Kokkos::Experimental;

template<class ValueType>
struct GenerateFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()() const {
    return static_cast<ValueType>(23);
  }
};

template <class ViewType, class AuxView, class MemberType>
struct FunctorA {
  AuxView m_viewOfDistances;
  ViewType m_view;
  int m_api_pick;
  int m_numFills;

  FunctorA(AuxView viewOfDistances, const ViewType view,
                   int apiPick, int numFills)
      : m_viewOfDistances(viewOfDistances),
        m_view(view),
        m_api_pick(apiPick),
        m_numFills(numFills) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const
  {
    using value_type = typename ViewType::value_type;

    const auto myRowIndex = member.league_rank();
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    if (m_api_pick == 0) {
      auto it = KE::generate_n(member, KE::begin(myRowView), m_numFills, GenerateFunctor<value_type>());
      const auto itDist                = KE::distance(KE::begin(myRowView), it);
      m_viewOfDistances(myRowIndex, 0) = itDist;
    }

    else if (m_api_pick == 1) {
      auto it           = KE::generate_n(member, myRowView, m_numFills, GenerateFunctor<value_type>());
      const auto itDist = KE::distance(KE::begin(myRowView), it);
      m_viewOfDistances(myRowIndex, 0) = itDist;
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols, std::size_t num_fills,
            int apiId) {

  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");

  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  // make view that will contain the computed distances
  // from begin(v) of the iterators returned by generate_n
  Kokkos::View<std::size_t**> computedDistances("cd", num_teams, 1);

  using functor_type =
      FunctorA<decltype(v), decltype(computedDistances),
                       team_member_type>;
  functor_type fnc(computedDistances, v, apiId, num_fills);
  Kokkos::parallel_for(policy, fnc);

  auto cd_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  computedDistances);
  for (std::size_t i = 0; i < cd_h.extent(0); ++i) {
    EXPECT_TRUE(cd_h(i, 0) == num_fills);
  }

  // check results
  auto v_h = create_host_space_copy(v);
  for (std::size_t i = 0; i < v_h.extent(0); ++i) {
    for (std::size_t j = 0; j < v_h.extent(1); ++j) {
      if (j < num_fills) {
        EXPECT_TRUE(v_h(i, j) == static_cast<ValueType>(23));
      } else {
        EXPECT_TRUE(v_h(i, j) == static_cast<ValueType>(0));
      }
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  // key = num of columns,
  // value = list of num of elements to generate
  using v_t                          = std::vector<int>;
  const std::map<int, v_t> scenarios = {{0, v_t{0}},
                                        {2, v_t{0, 1, 2}},
                                        {6, v_t{0, 1, 2, 5}},
                                        {13, v_t{0, 1, 2, 8, 11}}};

  for (int num_teams : team_sizes_to_test) {
    for (const auto& scenario : scenarios) {
      const std::size_t numCols = scenario.first;
      for (int numFills : scenario.second) {
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(num_teams, numCols, numFills, apiId);
        }
      }
    }
  }
}

TEST(std_algorithms_generate_n_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamGenerate_n
}  // namespace stdalgos
}  // namespace Test
