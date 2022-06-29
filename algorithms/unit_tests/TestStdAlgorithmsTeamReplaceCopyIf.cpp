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
namespace TeamReplaceCopyIf {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct IsGreaterThanValueFunctor {
  ValueType m_val;

  KOKKOS_INLINE_FUNCTION
  IsGreaterThanValueFunctor(ValueType val) : m_val(val) {}

  KOKKOS_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class ViewFromType, class ViewDestType, class MemberType,
          class ReplaceCopyIfUnaryOpType>
struct TestFunctorA {
  ViewFromType m_from_view;
  ViewDestType m_dest_view;
  int m_api_pick;

  TestFunctorA(const ViewFromType viewFrom, const ViewDestType viewDest,
               int apiPick)
      : m_from_view(viewFrom), m_dest_view(viewDest), m_api_pick(apiPick) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();

    auto myRowViewFrom =
        Kokkos::subview(m_from_view, myRowIndex, Kokkos::ALL());
    auto myRowViewDest =
        Kokkos::subview(m_dest_view, myRowIndex, Kokkos::ALL());

    if (m_api_pick == 0) {
      auto it = KE::replace_copy_if(
          member, KE::begin(myRowViewFrom), KE::end(myRowViewFrom),
          KE::begin(myRowViewDest), ReplaceCopyIfUnaryOpType(151), 1);
      (void)it;
    } else if (m_api_pick == 1) {
      auto it = KE::replace_copy_if(member, myRowViewFrom, myRowViewDest,
                                    ReplaceCopyIfUnaryOpType(151), 1);
      (void)it;
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t num_teams, std::size_t num_cols, int apiId) {
  /* description:
     randomly fill a rank-2 view with values between 0 and 523
     and then we run a team-level replace_copy_if where we replace copy
     the values that are greater than 151 with 1
     (note that these are purely arbitrary numbers)
   */

  // v constructed on memory space associated with default exespace
  auto v = create_view<ValueType>(Tag{}, num_teams, num_cols, "v");

  // v might not deep copyable so to modify it on the host
  auto v_dc   = create_deep_copyable_compatible_view_with_same_extent(v);
  auto v_dc_h = create_mirror_view(Kokkos::HostSpace(), v_dc);

  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(12371);
  Kokkos::fill_random(v_dc_h, pool, 0, 523);
  std::vector<std::size_t> rowIndOfTargetElements;
  std::vector<std::size_t> colIndOfTargetElements;
  for (std::size_t i = 0; i < v_dc_h.extent(0); ++i) {
    for (std::size_t j = 0; j < v_dc_h.extent(1); ++j) {
      if (v_dc_h(i, j) > static_cast<ValueType>(151)) {
        rowIndOfTargetElements.push_back(i);
        colIndOfTargetElements.push_back(j);
      }
    }
  }

  // copy to v_dc and then to v
  Kokkos::deep_copy(v_dc, v_dc_h);
  CopyFunctorRank2<decltype(v_dc), decltype(v)> F1(v_dc, v);
  Kokkos::parallel_for("copy", v.extent(0) * v.extent(1), F1);

  // launch kernel
  using space_t          = Kokkos::DefaultExecutionSpace;
  using policy_type      = Kokkos::TeamPolicy<space_t>;
  using team_member_type = typename policy_type::member_type;
  policy_type policy(num_teams, Kokkos::AUTO());

  // v2 is the destination view where we copy values to
  auto v2     = create_view<ValueType>(Tag{}, num_teams, num_cols, "v2");
  using bop_t = IsGreaterThanValueFunctor<ValueType>;
  using functor_type =
      TestFunctorA<decltype(v), decltype(v2), team_member_type, bop_t>;
  functor_type fnc(v, v2, apiId);
  Kokkos::parallel_for(policy, fnc);

  // check
  auto v_h  = create_host_space_copy(v);
  auto v2_h = create_host_space_copy(v2);
  for (std::size_t k = 0; k < rowIndOfTargetElements.size(); ++k) {
    EXPECT_TRUE(v_h(rowIndOfTargetElements[k], colIndOfTargetElements[k]) >
                static_cast<ValueType>(151));
    EXPECT_TRUE(v2_h(rowIndOfTargetElements[k], colIndOfTargetElements[k]) ==
                static_cast<ValueType>(1));
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

TEST(std_algorithms_replace_copy_if_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplaceCopyIf
}  // namespace stdalgos
}  // namespace Test
