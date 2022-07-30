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
namespace TeamReplaceIf {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct GreaterThanValueFunctor {
  ValueType m_val;

  KOKKOS_INLINE_FUNCTION
  GreaterThanValueFunctor(ValueType val) : m_val(val) {}

  KOKKOS_INLINE_FUNCTION
  bool operator()(ValueType val) const { return (val > m_val); }
};

template <class ViewType, class ValueType>
struct TestFunctorA {
  ViewType m_view;
  ValueType m_threshold;
  ValueType m_newVal;
  int m_apiPick;

  TestFunctorA(const ViewType view, ValueType threshold, ValueType newVal,
               int apiPick)
      : m_view(view),
        m_threshold(threshold),
        m_newVal(newVal),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    GreaterThanValueFunctor predicate(m_threshold);
    if (m_apiPick == 0) {
      KE::replace_if(member, KE::begin(myRowView), KE::end(myRowView),
                     predicate, m_newVal);
    } else if (m_apiPick == 1) {
      KE::replace_if(member, myRowView, predicate, m_newVal);
    }
  }
};

template <class Tag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values between 0 and 523
     and run a team-level replace_if where the values strictly greater
     than a threshold are replaced with a new value.
   */
  const auto threshold = static_cast<ValueType>(151);
  const auto newVal    = static_cast<ValueType>(1);

  //
  // prepare data
  //
  // construct in memory space associated with default exespace
  auto dataView = create_view<ValueType>(Tag{}, numTeams, numCols, "dataView");

  // dataView might not deep copyable (e.g. strided layout) so to fill it
  // we make a new view that is for sure deep copyable, modify it on the host
  // deep copy to device and then launch copy kernel to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  // randomly fill the view with values between 0 and 523
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(12371);
  Kokkos::fill_random(dataView_dc_h, pool, 0, 523);

  // copy to dataView_dc and then to dataView
  Kokkos::deep_copy(dataView_dc, dataView_dc_h);
  // use CTAD
  CopyFunctorRank2 F1(dataView_dc, dataView);
  Kokkos::parallel_for("copy", dataView.extent(0) * dataView.extent(1), F1);

  //
  // launch kokkos kernel
  //
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());
  // use CTAD for functor
  TestFunctorA fnc(dataView, threshold, newVal, apiId);
  Kokkos::parallel_for(policy, fnc);

  //
  // run cpp-std kernel
  //
  GreaterThanValueFunctor predicate(threshold);
  for (std::size_t i = 0; i < dataView_dc_h.extent(0); ++i) {
    auto thisRow = Kokkos::subview(dataView_dc_h, i, Kokkos::ALL());
    std::replace_if(KE::begin(thisRow), KE::end(thisRow), predicate, newVal);
  }

  //
  // check
  //
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  expect_equal_host_views(dataView_dc_h, dataViewAfterOp_h);
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  for (int numTeams : team_sizes_to_test) {
    for (const auto& numCols : {0, 1, 2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<Tag, ValueType>(numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_replace_if_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamReplaceIf
}  // namespace stdalgos
}  // namespace Test
