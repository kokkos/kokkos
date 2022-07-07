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
namespace TeamTransform {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct TimesTwoUnaryOp
{
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType & val) const {
    return val * static_cast<ValueType>(2);
  }
};

template <class ValueType>
struct AddValuesBinaryOp
{
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType & a, const ValueType & b) const {
    return a + b;
  }
};

template <class ViewFromType, class ViewDestType, class MemberType>
struct TestFunctorA {
  ViewFromType m_from_view;
  ViewDestType m_dest_view;
  int m_api_pick;

  TestFunctorA(const ViewFromType viewFrom,
	       const ViewDestType viewDest,
               int apiPick)
      : m_from_view(viewFrom), m_dest_view(viewDest), m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom =
        Kokkos::subview(m_from_view, myRowIndex, Kokkos::ALL());
    auto myRowViewDest =
        Kokkos::subview(m_dest_view, myRowIndex, Kokkos::ALL());

    using value_type = typename ViewFromType::value_type;
    if (m_api_pick == 0)
    {
      auto it = KE::transform(member,
			      KE::begin(myRowViewFrom), KE::end(myRowViewFrom),
			      KE::begin(myRowViewDest),
			      TimesTwoUnaryOp<value_type>());
      (void) it;
    } else if (m_api_pick == 1)
    {
      auto it = KE::transform(member, myRowViewFrom, myRowViewDest,
			      TimesTwoUnaryOp<value_type>());
      (void) it;
    }
  }
};

template <
  class ViewFromType1, class ViewFromType2,
  class ViewDestType, class MemberType
  >
struct TestFunctorB {
  ViewFromType1 m_from_view1;
  ViewFromType2 m_from_view2;
  ViewDestType m_dest_view;
  int m_api_pick;

  TestFunctorB(const ViewFromType1 viewFrom1,
	       const ViewFromType2 viewFrom2,
	       const ViewDestType viewDest,
               int apiPick)
      : m_from_view1(viewFrom1), m_from_view2(viewFrom2),
	m_dest_view(viewDest), m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const
  {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom1 =
        Kokkos::subview(m_from_view1, myRowIndex, Kokkos::ALL());
    auto myRowViewFrom2 =
        Kokkos::subview(m_from_view2, myRowIndex, Kokkos::ALL());
    auto myRowViewDest =
        Kokkos::subview(m_dest_view, myRowIndex, Kokkos::ALL());

    using value_type = typename ViewFromType1::value_type;
    if (m_api_pick == 0)
    {
      auto it = KE::transform(member,
			      KE::begin(myRowViewFrom1), KE::end(myRowViewFrom1),
			      KE::begin(myRowViewFrom2), KE::begin(myRowViewDest),
			      AddValuesBinaryOp<value_type>());
      (void) it;
    } else if (m_api_pick == 1)
    {
      auto it = KE::transform(member, myRowViewFrom1, myRowViewFrom2,
			      myRowViewDest,
			      AddValuesBinaryOp<value_type>());
      (void) it;
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols,
	    int apiId, int opSwitch)
{
  /* description: */

  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> pool(12371);

  // v constructed on memory space associated with default exespace
  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");
  Kokkos::fill_random(v, pool, 0, 15);

  // launch kernel
  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  if (opSwitch == 0)
  {
    auto v2 = create_view<ValueType>(Tag{}, num_teams, num_cols, "v2");
    using functor_type = TestFunctorA<decltype(v), decltype(v2), team_member_type>;
    functor_type fnc(v, v2, apiId);
    Kokkos::parallel_for(policy, fnc);

    // check
    auto v_h = create_host_space_copy(v);
    auto v2_h = create_host_space_copy(v2);
    for (std::size_t i = 0; i < v2_h.extent(0); ++i) {
      for (std::size_t j = 0; j < v2_h.extent(1); ++j) {
	EXPECT_TRUE( v2_h(i,j) == static_cast<ValueType>(2)*v_h(i,j) );
      }
    }
  }
  else
  {
    auto v2 = create_view<ValueType>(Tag{}, num_teams, num_cols, "v2");
    Kokkos::fill_random(v2, pool, 0, 15);

    auto v3 = create_view<ValueType>(Tag{}, num_teams, num_cols, "v3");
    using functor_type = TestFunctorB<decltype(v), decltype(v2),
				      decltype(v3), team_member_type>;
    functor_type fnc(v, v2, v3, apiId);
    Kokkos::parallel_for(policy, fnc);

    // check
    auto v_h = create_host_space_copy(v);
    auto v2_h = create_host_space_copy(v2);
    auto v3_h = create_host_space_copy(v3);
    for (std::size_t i = 0; i < v3_h.extent(0); ++i) {
      for (std::size_t j = 0; j < v3_h.extent(1); ++j) {
	EXPECT_TRUE( v3_h(i,j) == v_h(i,j) + v2_h(i,j) );
      }
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios(int opSwitch) {
  for (int num_teams : team_sizes_to_test) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 51153}) {
      for (int apiId : {0, 1}) {
	test_A<Tag, ValueType>(num_teams, numCols, apiId, opSwitch);
      }
    }
  }
}

TEST(std_algorithms_transform_team_test, test_unary_op) {
  run_all_scenarios<DynamicTag, double>(0);
  run_all_scenarios<StridedTwoRowsTag, int>(0);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(0);
}

TEST(std_algorithms_transform_team_test, test_binary_op) {
  run_all_scenarios<DynamicTag, double>(1);
  run_all_scenarios<StridedTwoRowsTag, int>(1);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(1);
}

}  // namespace TeamTransform
}  // namespace stdalgos
}  // namespace Test
